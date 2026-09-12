# ============================================================================
# DEPRECATED — AWS account 857194222592, decommissioned 2026-09-12.
# Do NOT apply. The live module is deploy/aws/terraform-new-account/.
# See DEPRECATED.md in this directory (includes a leaked-key disclosure).
# ============================================================================
provider "aws" {
  region  = var.aws_region
  profile = var.aws_profile

  default_tags {
    tags = {
      Project   = "magik"
      ManagedBy = "terraform"
    }
  }
}

data "aws_caller_identity" "current" {}
