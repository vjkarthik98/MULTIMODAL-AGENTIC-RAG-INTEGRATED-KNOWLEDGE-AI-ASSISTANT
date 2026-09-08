# =============================================================================
# GitHub OIDC provider + the role cd.yml assumes (magik-deploy-role)
# =============================================================================
#
# Brand-new account has zero IAM providers/roles (confirmed via
# `aws iam list-open-id-connect-providers` / `list-roles`), so both are created
# fresh here — this repo's existing deploy/aws/iam/github-oidc-trust-policy.json
# is reused as the SOURCE OF TRUTH for the sub-claim patterns (both the
# ID-qualified and legacy forms, per that file's own hard-won comment about why
# both must be present), just re-expressed as Terraform so the account ID is
# never hand-typed again.

data "tls_certificate" "github_oidc" {
  url = "https://token.actions.githubusercontent.com"
}

resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.github_oidc.certificates[0].sha1_fingerprint]
}

locals {
  # Both forms, both environments, both required — see
  # deploy/aws/iam/github-oidc-trust-policy.json's _comment for the full
  # 2026-08-07 incident writeup on why a single form silently breaks.
  github_oidc_subs = [
    "repo:${var.github_owner}@${var.github_owner_id}/${var.github_repo}@${var.github_repo_id}:environment:production",
    "repo:${var.github_owner}@${var.github_owner_id}/${var.github_repo}@${var.github_repo_id}:environment:staging",
    "repo:${var.github_owner}@${var.github_owner_id}/${var.github_repo}@${var.github_repo_id}:ref:refs/tags/*",
    "repo:${var.github_owner}/${var.github_repo}:environment:production",
    "repo:${var.github_owner}/${var.github_repo}:environment:staging",
    "repo:${var.github_owner}/${var.github_repo}:ref:refs/tags/*",
  ]
}

data "aws_iam_policy_document" "deploy_role_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = local.github_oidc_subs
    }
  }
}

resource "aws_iam_role" "magik_deploy" {
  name               = "magik-deploy-role"
  assume_role_policy = data.aws_iam_policy_document.deploy_role_trust.json
}

# Permissions side of this role: the original repo's own README noted this was
# "managed directly in AWS... not captured in-repo" — a real gap, closed here.
# Scoped to exactly what cd.yml's SSM-based deploy needs, against the two
# specific instances (dynamic ARN references, never a hardcoded instance ID —
# this is the exact fragility that broke the Lambda permission JSONs last time
# an instance was replaced).
data "aws_iam_policy_document" "deploy_permissions" {
  statement {
    sid       = "DescribeInstances"
    effect    = "Allow"
    actions   = ["ec2:DescribeInstances"]
    resources = ["*"] # DescribeInstances cannot be resource-scoped
  }

  statement {
    sid     = "StartStopTaggedInstances"
    effect  = "Allow"
    actions = ["ec2:StartInstances", "ec2:StopInstances"]
    resources = concat(
      [aws_instance.production.arn],
      var.create_staging ? [aws_instance.staging[0].arn] : [],
    )
  }

  statement {
    sid    = "SsmDeploy"
    effect = "Allow"
    actions = [
      "ssm:SendCommand",
      "ssm:GetCommandInvocation",
      "ssm:ListCommands",
      "ssm:ListCommandInvocations",
    ]
    resources = ["*"]
  }

  statement {
    sid       = "SsmDescribeInstanceInfo"
    effect    = "Allow"
    actions   = ["ssm:DescribeInstanceInformation"]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "magik_deploy_permissions" {
  name   = "magik-deploy-permissions"
  role   = aws_iam_role.magik_deploy.id
  policy = data.aws_iam_policy_document.deploy_permissions.json
}

# =============================================================================
# EC2 instance profile (attached to both boxes — one shared role, matching the
# original "nothing in this file is prod-specific" note in
# ec2-instance-profile-permissions.json)
# =============================================================================

data "aws_iam_policy_document" "ec2_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "magik_ec2" {
  name               = "magik-ec2-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume.json
}

resource "aws_iam_role_policy_attachment" "ssm_managed_core" {
  role       = aws_iam_role.magik_ec2.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

# Read-only, tightly scoped to exactly the 9 app secrets + GHCR PAT + 2
# monitoring secrets under /magik/* — never a wildcard ssm:GetParameter, same
# invariant the original ec2-instance-profile-permissions.json documented.
data "aws_iam_policy_document" "ec2_ssm_secrets" {
  statement {
    sid       = "ReadGhcrPullToken"
    effect    = "Allow"
    actions   = ["ssm:GetParameter"]
    resources = ["arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter/magik/ghcr_pat"]
  }

  statement {
    sid     = "ReadAppSecretsAtDeployTime"
    effect  = "Allow"
    actions = ["ssm:GetParameter"]
    resources = [
      for name in [
        "google_client_secret", "smtp_password", "secret_key", "jwt_secret_key",
        "mongo_uri", "qdrant_api_key", "redis_token", "hf_token", "tavily_api_key",
      ] : "arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter/magik/${name}"
    ]
  }

  statement {
    sid     = "ReadMonitoringSecretsAtDeployTime"
    effect  = "Allow"
    actions = ["ssm:GetParameter"]
    resources = [
      for name in ["grafana_admin_password", "ntfy_webhook_url"] :
      "arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter/magik/${name}"
    ]
  }
}

resource "aws_iam_role_policy" "magik_ec2_ssm_secrets" {
  name   = "magik-ec2-ssm-secrets"
  role   = aws_iam_role.magik_ec2.id
  policy = data.aws_iam_policy_document.ec2_ssm_secrets.json
}

resource "aws_iam_instance_profile" "magik_ec2" {
  name = "magik-ec2-instance-profile"
  role = aws_iam_role.magik_ec2.name
}
