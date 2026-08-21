# Naming moved since the last build (was "...GPU PyTorch...(Ubuntu 22.04)...") —
# confirmed live via `aws ec2 describe-images` on 2026-08-21. No AMI ID pinned
# (never was, in the original setup either); most_recent + a version-agnostic
# name pattern so this doesn't silently go stale again.
data "aws_ami" "deep_learning" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning OSS Nvidia Driver AMI GPU PyTorch*(Ubuntu 24.04)*"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# =============================================================================
# Production
# =============================================================================

resource "aws_instance" "production" {
  ami                    = data.aws_ami.deep_learning.id
  instance_type          = var.instance_type
  availability_zone      = var.availability_zone
  subnet_id              = aws_subnet.magik_public.id
  vpc_security_group_ids = [aws_security_group.production.id]
  key_name               = aws_key_pair.magik.key_name
  iam_instance_profile   = aws_iam_instance_profile.magik_ec2.name

  # Root/boot volume — 100GiB per spec: OS, Docker + images, git checkout,
  # /opt/magik/{data,logs,.env}. Deliberately NOT where models live.
  root_block_device {
    volume_size           = var.root_volume_size_gb
    volume_type           = "gp3"
    delete_on_termination = false # a stop/terminate must never silently drop this
    tags                  = { Name = "magik-prod-root" }
  }

  tags = {
    Name        = "magik-prod"
    Environment = "production"
  }

  lifecycle {
    ignore_changes = [ami] # don't force-replace the box on every AMI refresh
  }
}

# Dedicated model-cache volume — 100GiB per spec, mounted at
# /opt/magik/.hf_cache. Kept as its own aws_ebs_volume (not part of the root
# device) so it can be snapshotted/cloned to staging independently and is
# never at risk from any root-volume-scoped operation.
resource "aws_ebs_volume" "prod_models" {
  availability_zone = var.availability_zone
  size              = var.model_volume_size_gb
  type              = "gp3"

  tags = { Name = "magik-prod-models" }

  lifecycle {
    prevent_destroy = true # models take ~20GB/hours to redownload — never destroy by accident
  }
}

resource "aws_volume_attachment" "prod_models" {
  device_name = "/dev/sdf"
  volume_id   = aws_ebs_volume.prod_models.id
  instance_id = aws_instance.production.id
}

resource "aws_eip" "production" {
  domain = "vpc"
  tags   = { Name = "magik-prod-eip" }
}

resource "aws_eip_association" "production" {
  instance_id   = aws_instance.production.id
  allocation_id = aws_eip.production.id
}

# =============================================================================
# Staging
# =============================================================================

resource "aws_instance" "staging" {
  count = var.create_staging ? 1 : 0

  ami                    = data.aws_ami.deep_learning.id
  instance_type          = var.instance_type
  availability_zone      = var.availability_zone
  subnet_id              = aws_subnet.magik_public.id
  vpc_security_group_ids = [aws_security_group.staging.id]
  key_name               = aws_key_pair.magik.key_name
  iam_instance_profile   = aws_iam_instance_profile.magik_ec2.name

  # NOTE: aws_instance's inline root_block_device cannot clone from a
  # snapshot (that argument doesn't exist for it — only the top-level
  # aws_ebs_volume resource supports snapshot_id). Staging's root volume is
  # therefore always blank; only its dedicated MODEL volume below can be
  # cloned from production's snapshot. Staging's BM25 index/data get rebuilt
  # independently on the blank root volume during staging bootstrap.
  root_block_device {
    volume_size           = var.root_volume_size_gb
    volume_type           = "gp3"
    delete_on_termination = false
    tags                  = { Name = "magik-staging-root" }
  }

  tags = {
    Name        = "magik-staging"
    Environment = "staging"
  }

  lifecycle {
    ignore_changes = [ami]
  }
}

# Staging's model volume: blank on the very first apply
# (staging_model_snapshot_id = null); after production's models are downloaded
# and snapshotted, set the variable and re-apply — Terraform will recreate
# just this volume from the snapshot (models arrive pre-populated, no second
# ~20GB download).
resource "aws_ebs_volume" "staging_models" {
  count = var.create_staging ? 1 : 0

  availability_zone = var.availability_zone
  size              = var.model_volume_size_gb
  type              = "gp3"
  snapshot_id       = var.staging_model_snapshot_id

  tags = { Name = "magik-staging-models" }
}

resource "aws_volume_attachment" "staging_models" {
  count = var.create_staging ? 1 : 0

  device_name = "/dev/sdf"
  volume_id   = aws_ebs_volume.staging_models[0].id
  instance_id = aws_instance.staging[0].id
}

# No Elastic IP for staging on purpose — nothing (DNS, the wake gateway)
# depends on a stable staging address; cd.yml resolves it fresh by tag every
# run, and staging is never reached over the public internet at all.
