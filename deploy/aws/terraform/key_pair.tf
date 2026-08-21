# Break-glass SSH key. The CD pipeline itself never uses this (SSM only) —
# this exists solely for manual admin access (initial bootstrap: mounting EBS
# volumes, downloading models, registering the self-hosted runner; and later,
# whenever SSM itself is the thing that's broken).
resource "tls_private_key" "magik" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "magik" {
  key_name   = "magik-admin-key"
  public_key = tls_private_key.magik.public_key_openssh
}

# Private key saved locally, 0600. NEVER commit this file — it's gitignored.
# Back it up somewhere safe (password manager) immediately after first apply;
# if lost, the only recovery is EC2 Instance Connect / SSM Session Manager
# (both still work without this key) or generating a new key pair entirely.
resource "local_sensitive_file" "magik_private_key" {
  content         = tls_private_key.magik.private_key_pem
  filename        = "${path.module}/magik-admin-key.pem"
  file_permission = "0600"
}
