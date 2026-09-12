# Fresh AWS account (266901698137) has NO default VPC in us-east-1 at all —
# confirmed via `aws ec2 describe-vpcs` returning zero results. Everything
# below is new, not a reuse of a pre-existing default VPC as the original
# (deleted) setup implicitly assumed.
#
# TWO public subnets, two AZs — not by choice. The original design here was a
# single public subnet in a single AZ, matching the project's "single
# resource-constrained host, keep it simple" cost philosophy (see
# monitoring/slo.md's reasoning for the same pattern). GPU capacity forced the
# second one; see the magik_public_1b comment below. Both subnets are public
# and share one route table, so this is still not a multi-AZ HA design — it is
# a single-AZ design that had to move AZ twice. No NAT Gateway — staging has
# zero inbound but still needs outbound (SSM agent, docker pull, apt), and a
# public subnet + IGW gives it that for free. A NAT Gateway would cost
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

# Second subnet, second AZ — NOT part of the original single-AZ design.
# g6e.xlarge hit InsufficientInstanceCapacity TWICE in this account during
# bring-up (2026-09-08): first in us-east-1a, then again in us-east-1b after
# moving there — production ended up in us-east-1c on the second retry.
# Resource address kept as "magik_public_1b" (not renamed to _1c) to avoid
# an unnecessary destroy/recreate churn on a working subnet; its actual AZ is
# whatever var.production_availability_zone currently says (us-east-1c as of
# the second move). EBS volumes are AZ-locked, so production's model volume
# moves with it. Staging later had to move here too — it could not get
# g6e.xlarge capacity in ANY AZ for several days (2026-09-09 → 2026-09-12) and
# finally launched in us-east-1c, so both GPU boxes now share this subnet.
# Uptime Kuma (t4g.micro, no capacity issue anywhere) is the only thing still
# on the original var.availability_zone subnet.
resource "aws_subnet" "magik_public_1b" {
  vpc_id                  = aws_vpc.magik.id
  cidr_block              = "10.0.3.0/24" # real CIDR of the us-east-1c subnet (10.0.2.0/24 was the abandoned 1b one)
  availability_zone       = var.production_availability_zone
  map_public_ip_on_launch = true

  tags = { Name = "magik-public-1c" }
}

resource "aws_route_table_association" "magik_public_1b" {
  subnet_id      = aws_subnet.magik_public_1b.id
  route_table_id = aws_route_table.magik_public.id
}
