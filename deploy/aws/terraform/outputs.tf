output "production_instance_id" {
  value = aws_instance.production.id
}

output "production_public_ip" {
  description = "The Elastic IP — point the magik.vk-ai.online GoDaddy A record at this."
  value       = aws_eip.production.public_ip
}

output "production_instance_arn" {
  description = "Feed this into deploy/aws/iam/lambda-*-permissions.json before re-running deploy_lambdas.sh."
  value       = aws_instance.production.arn
}

output "staging_instance_id" {
  value = var.create_staging ? aws_instance.staging[0].id : null
}

output "staging_instance_arn" {
  value = var.create_staging ? aws_instance.staging[0].arn : null
}

output "magik_deploy_role_arn" {
  description = "Set this as the AWS_DEPLOY_ROLE_ARN GitHub Actions repo variable."
  value       = aws_iam_role.magik_deploy.arn
}

output "prod_models_volume_id" {
  description = "Snapshot this (aws ec2 create-snapshot) after the first model download, then feed the snapshot ID into staging_model_snapshot_id."
  value       = aws_ebs_volume.prod_models.id
}

output "ssh_private_key_path" {
  description = "Break-glass SSH key — back this up outside the repo immediately."
  value       = local_sensitive_file.magik_private_key.filename
}

output "github_oidc_provider_arn" {
  value = aws_iam_openid_connect_provider.github.arn
}
