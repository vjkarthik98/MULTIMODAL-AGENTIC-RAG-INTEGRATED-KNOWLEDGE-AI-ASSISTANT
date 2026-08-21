# Private, encrypted, versioned bucket for manual Terraform state backups.
# NOT used as a remote backend (this project deliberately stays on local
# state — see versions.tf's comment: single-operator rebuild, not a team
# workflow) — just a durable, off-laptop copy so a lost/corrupted laptop
# doesn't repeat the "no record of what existed" problem this whole rebuild
# was fixing. The state file contains real sensitive material in plaintext
# (the tls_private_key resource's SSH private key, IPs, resource IDs), so
# this bucket blocks all public access and encrypts at rest by default.
#
# Manual backup step after any apply that changes state:
#   aws s3 cp deploy/aws/terraform/terraform.tfstate \
#     s3://magik-terraform-state-857194222592/terraform.tfstate
resource "aws_s3_bucket" "terraform_state_backup" {
  bucket = "magik-terraform-state-857194222592"

  tags = { Name = "magik-terraform-state-backup" }
}

resource "aws_s3_bucket_versioning" "terraform_state_backup" {
  bucket = aws_s3_bucket.terraform_state_backup.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "terraform_state_backup" {
  bucket = aws_s3_bucket.terraform_state_backup.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "terraform_state_backup" {
  bucket = aws_s3_bucket.terraform_state_backup.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
