# Uptime Kuma host — small, separate, always-on box per
# monitoring/uptime-kuma/README.md. Deliberately NOT the g6e.xlarge GPU boxes
# (those sleep by design; a status page hosted there would go dark exactly
# when it's most useful to check) and deliberately NOT reusing magik_ec2's IAM
# role (that role can read all 9 app + 2 monitoring SSM secrets — this box
# needs none of them, so it gets its own minimal SSM-only role instead).
#
# create_uptime_kuma defaults false: this is a real, separate recurring AWS
# cost (~$3-8/mo), provisioned deliberately via `terraform apply` with the
# variable explicitly flipped, never as a side effect of any other apply.

data "aws_ami" "small_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_security_group" "uptime_kuma" {
  count = var.create_uptime_kuma ? 1 : 0

  name        = "magik-uptime-kuma-sg"
  description = "MAGIK uptime-kuma - HTTP/HTTPS only, no SSH (SSM Session Manager instead)"
  vpc_id      = aws_vpc.magik.id

  ingress {
    description = "HTTP (Caddy redirects to HTTPS, ACME challenge)"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS (Caddy: Kuma push endpoint + status page + basic_auth dashboard)"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "magik-uptime-kuma-sg" }
}

data "aws_iam_policy_document" "uptime_kuma_ec2_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

# SSM Session Manager access only — no ssm:GetParameter on any /magik/*
# secret. This box never needs an app secret; it only runs Kuma + Caddy.
resource "aws_iam_role" "uptime_kuma_ec2" {
  count = var.create_uptime_kuma ? 1 : 0

  name               = "magik-uptime-kuma-ec2-role"
  assume_role_policy = data.aws_iam_policy_document.uptime_kuma_ec2_assume.json
}

resource "aws_iam_role_policy_attachment" "uptime_kuma_ssm_managed_core" {
  count = var.create_uptime_kuma ? 1 : 0

  role       = aws_iam_role.uptime_kuma_ec2[0].name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "uptime_kuma" {
  count = var.create_uptime_kuma ? 1 : 0

  name = "magik-uptime-kuma-instance-profile"
  role = aws_iam_role.uptime_kuma_ec2[0].name
}

resource "aws_instance" "uptime_kuma" {
  count = var.create_uptime_kuma ? 1 : 0

  ami                    = data.aws_ami.small_linux.id
  instance_type          = var.uptime_kuma_instance_type
  availability_zone      = var.availability_zone
  subnet_id              = aws_subnet.magik_public.id
  vpc_security_group_ids = [aws_security_group.uptime_kuma[0].id]
  iam_instance_profile   = aws_iam_instance_profile.uptime_kuma[0].name
  # No key_name on purpose — SSM only, matching staging's zero-SSH pattern.

  root_block_device {
    volume_size           = 30 # AMI's snapshot requires >= 30GB; OS + Docker + Kuma's small sqlite DB — no models, no app data
    volume_type           = "gp3"
    delete_on_termination = true # nothing irreplaceable lives on this box; Kuma's monitor config is trivial to recreate
    tags                  = { Name = "magik-uptime-kuma-root" }
  }

  tags = {
    Name        = "magik-uptime-kuma"
    Environment = "monitoring"
  }

  lifecycle {
    ignore_changes = [ami]
  }
}

resource "aws_eip" "uptime_kuma" {
  count  = var.create_uptime_kuma ? 1 : 0
  domain = "vpc"
  tags   = { Name = "magik-uptime-kuma-eip" }
}

resource "aws_eip_association" "uptime_kuma" {
  count         = var.create_uptime_kuma ? 1 : 0
  instance_id   = aws_instance.uptime_kuma[0].id
  allocation_id = aws_eip.uptime_kuma[0].id
}
