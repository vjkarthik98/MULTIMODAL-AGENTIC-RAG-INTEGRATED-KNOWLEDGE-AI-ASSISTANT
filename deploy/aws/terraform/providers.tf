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
