variable "aws_region" {
  description = "AWS region for all MAGIK infrastructure."
  type        = string
  default     = "us-east-1"
}

variable "aws_profile" {
  description = "Named AWS CLI profile to use for the local operator running Terraform (NOT what GitHub Actions uses — that's the separate OIDC deploy role created by this config)."
  type        = string
  default     = "magik-admin"
}

variable "availability_zone" {
  description = "Single AZ hosting both boxes — matches the original setup and keeps EBS snapshot->volume cloning trivial (no cross-AZ copy needed)."
  type        = string
  default     = "us-east-1a"
}

variable "admin_ssh_cidr" {
  description = "CIDR allowed to SSH (port 22) into the production box for break-glass admin access. Never 0.0.0.0/0. Find yours with: curl -s https://checkip.amazonaws.com"
  type        = string
  # No default on purpose — must be supplied in terraform.tfvars.
}

variable "github_owner" {
  description = "GitHub org/user that owns the repo (legacy-form OIDC sub claim)."
  type        = string
  default     = "vjkarthik98"
}

variable "github_owner_id" {
  description = "GitHub's numeric ID for github_owner (ID-qualified OIDC sub claim, immutable even if the account is renamed). Verify with: gh api users/<owner> --jq .id"
  type        = string
  default     = "218363746"
}

variable "github_repo" {
  description = "Repo name (legacy-form OIDC sub claim)."
  type        = string
  default     = "MULTIMODAL-AGENTIC-RAG-INTEGRATED-KNOWLEDGE-AI-ASSISTANT"
}

variable "github_repo_id" {
  description = "GitHub's numeric ID for this repo (ID-qualified OIDC sub claim). Verify with: gh api repos/<owner>/<repo> --jq .id"
  type        = string
  default     = "1180642349"
}

variable "root_volume_size_gb" {
  description = "Root/boot EBS volume size (OS, Docker, git checkout, /opt/magik/{data,logs,.env}) — per explicit spec, 100GiB, separate from the model volume."
  type        = number
  default     = 100
}

variable "model_volume_size_gb" {
  description = "Dedicated EBS volume size for the HF model cache (/opt/magik/.hf_cache) — per explicit spec, 100GiB, kept separate so model weights are never at risk from root-volume operations."
  type        = number
  default     = 100
}

variable "instance_type" {
  description = "EC2 instance type for both boxes — g6e.xlarge (1x NVIDIA L40S, 48GB VRAM), matching the last known-good production hardware generation."
  type        = string
  default     = "g6e.xlarge"
}

variable "create_staging" {
  description = "Whether to create the staging instance yet. Leave false for the first apply (production-only bring-up); flip to true once production's model volume has been snapshotted, so staging launches with pre-populated models instead of paying for an idle box in the meantime."
  type        = bool
  default     = false
}

variable "staging_model_snapshot_id" {
  description = <<-EOT
    EBS snapshot ID to clone staging's model volume from (production's populated
    .hf_cache volume, snapshotted after the first model download). Leave null for
    the FIRST apply, which brings up production only — set this and re-apply once
    the snapshot exists to bring up staging with pre-populated models.
  EOT
  type        = string
  default     = null
}

variable "create_uptime_kuma" {
  description = "Whether to create the small, separate, always-on Uptime Kuma host (see monitoring/uptime-kuma/README.md). A real recurring cost (~$3-8/mo) — flip to true deliberately, in its own apply."
  type        = bool
  default     = false
}

variable "uptime_kuma_instance_type" {
  description = "Instance type for the Uptime Kuma host — deliberately tiny, it only runs Kuma + Caddy and receives passive pushes."
  type        = string
  default     = "t3.micro"
}

