# Production: 22 (restricted to admin_ssh_cidr, break-glass only — cd.yml uses
# SSM, never SSH), 80/443 (Caddy, public app + Grafana behind basic_auth).
resource "aws_security_group" "production" {
  name        = "magik-prod-sg"
  description = "MAGIK production - SSH (admin only), HTTP/HTTPS (public app via Caddy)"
  vpc_id      = aws_vpc.magik.id

  ingress {
    description = "SSH (break-glass admin only)"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.admin_ssh_cidr]
  }

  ingress {
    description = "HTTP (Caddy redirects to HTTPS, ACME challenge)"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS (Caddy: app + /grafana/)"
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

  tags = { Name = "magik-prod-sg" }
}

# Staging: zero inbound rules, by design — matches the original
# sg-0bbfb2a5141b55e87. Reached exclusively via SSM (AWS-RunShellScript), never
# HTTP/SSH from the internet. Outbound only (SSM agent, docker pull, apt, model
# download if ever needed independently of the snapshot clone).
resource "aws_security_group" "staging" {
  name        = "magik-staging-sg"
  description = "MAGIK staging - zero inbound, SSM-only access"
  vpc_id      = aws_vpc.magik.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "magik-staging-sg" }
}
