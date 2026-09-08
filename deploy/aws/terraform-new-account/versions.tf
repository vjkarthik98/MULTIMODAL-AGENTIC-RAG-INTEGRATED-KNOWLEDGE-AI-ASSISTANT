terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.4"
    }
  }

  # Local backend deliberately: this is a single-operator rebuild, not a team
  # workflow. terraform.tfstate must NOT be committed (see .gitignore) — it
  # contains the VPC/instance/EIP IDs and, transiently, the SSH private key
  # material from tls_private_key. Back it up after every apply that changes
  # state (private, versioned, encrypted bucket — see state_backup.tf):
  #   aws s3 cp terraform.tfstate s3://magik-terraform-state-857194222592/terraform.tfstate
  # so a lost laptop doesn't repeat the "deleted everything, no record of
  # what existed" problem this whole rebuild is fixing.
}
