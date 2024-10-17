# Outputs for AI Model Serving Platform

output "cluster_name" {
  description = "Name of the EKS cluster"
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "cluster_iam_role_arn" {
  description = "IAM role ARN associated with EKS cluster"
  value       = module.eks.cluster_iam_role_arn
}

output "node_groups" {
  description = "EKS node groups"
  value       = module.eks.node_groups
}

output "vpc_id" {
  description = "ID of the VPC"
  value       = module.networking.vpc_id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = module.networking.vpc_cidr_block
}

output "private_subnet_ids" {
  description = "List of IDs of private subnets"
  value       = module.networking.private_subnet_ids
}

output "public_subnet_ids" {
  description = "List of IDs of public subnets"
  value       = module.networking.public_subnet_ids
}

output "database_endpoint" {
  description = "RDS instance endpoint"
  value       = module.managed_services.database_endpoint
  sensitive   = true
}

output "database_port" {
  description = "RDS instance port"
  value       = module.managed_services.database_port
}

output "redis_endpoint" {
  description = "ElastiCache Redis endpoint"
  value       = module.managed_services.redis_endpoint
}

output "redis_port" {
  description = "ElastiCache Redis port"
  value       = module.managed_services.redis_port
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket for model artifacts"
  value       = module.managed_services.s3_bucket_name
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket for model artifacts"
  value       = module.managed_services.s3_bucket_arn
}

