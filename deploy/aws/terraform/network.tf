# Fresh AWS account (857194222592) has NO default VPC in us-east-1 at all —
# confirmed via `aws ec2 describe-vpcs` returning zero results. Everything
# below is new, not a reuse of a pre-existing default VPC as the original
# (deleted) setup implicitly assumed.
#
# Single public subnet, single AZ: matches the project's existing "single
# resource-constrained host, keep it simple" cost philosophy (see
# monitoring/slo.md's reasoning for the same pattern). No NAT Gateway — staging
# has zero inbound but still needs outbound (SSM agent, docker pull, apt), and
# a public subnet + IGW gives it that for free. A NAT Gateway would cost
# ~$32/mo doing nothing but outbound routing that an IGW already provides here.

resource "aws_vpc" "magik" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = "magik-vpc" }
}

resource "aws_internet_gateway" "magik" {
  vpc_id = aws_vpc.magik.id
  tags   = { Name = "magik-igw" }
}

resource "aws_subnet" "magik_public" {
  vpc_id                  = aws_vpc.magik.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = var.availability_zone
  map_public_ip_on_launch = true

  tags = { Name = "magik-public" }
}

resource "aws_route_table" "magik_public" {
  vpc_id = aws_vpc.magik.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.magik.id
  }

  tags = { Name = "magik-public-rt" }
}

resource "aws_route_table_association" "magik_public" {
  subnet_id      = aws_subnet.magik_public.id
  route_table_id = aws_route_table.magik_public.id
}
