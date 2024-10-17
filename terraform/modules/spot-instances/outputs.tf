# Outputs for Spot Instances Module

output "spot_fleet_id" {
  description = "ID of the spot fleet request"
  value       = var.enable_spot_fleet ? aws_spot_fleet_request.batch_processing[0].id : null
}

output "spot_asg_name" {
  description = "Name of the spot instances auto scaling group"
  value       = var.enable_spot_asg ? aws_autoscaling_group.spot_batch_processing[0].name : null
}

output "spot_asg_arn" {
  description = "ARN of the spot instances auto scaling group"
  value       = var.enable_spot_asg ? aws_autoscaling_group.spot_batch_processing[0].arn : null
}

output "launch_template_id" {
  description = "ID of the launch template for spot instances"
  value       = aws_launch_template.spot_batch_processing.id
}

output "launch_template_latest_version" {
  description = "Latest version of the launch template"
  value       = aws_launch_template.spot_batch_processing.latest_version
}

output "security_group_id" {
  description = "ID of the security group for spot instances"
  value       = aws_security_group.spot_instances.id
}

output "iam_role_arn" {
  description = "ARN of the IAM role for spot instances"
  value       = aws_iam_role.spot_instance_role.arn
}

output "iam_instance_profile_name" {
  description = "Name of the IAM instance profile for spot instances"
  value       = aws_iam_instance_profile.spot_instance_profile.name
}

output "spot_fleet_role_arn" {
  description = "ARN of the spot fleet IAM role"
  value       = var.enable_spot_fleet ? aws_iam_role.spot_fleet_role[0].arn : null
}

output "cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group for spot instances"
  value       = aws_cloudwatch_log_group.spot_instances.name
}

output "cloudwatch_log_group_arn" {
  description = "ARN of the CloudWatch log group for spot instances"
  value       = aws_cloudwatch_log_group.spot_instances.arn
}

output "interruption_queue_url" {
  description = "URL of the SQS queue for spot interruption notifications"
  value       = var.enable_interruption_handler ? aws_sqs_queue.spot_interruption[0].url : null
}

output "interruption_queue_arn" {
  description = "ARN of the SQS queue for spot interruption notifications"
  value       = var.enable_interruption_handler ? aws_sqs_queue.spot_interruption[0].arn : null
}

# Cost optimization outputs
output "estimated_monthly_savings" {
  description = "Estimated monthly savings compared to on-demand instances"
  value = {
    description = "Estimated savings based on 70% spot discount"
    calculation = "Assumes 70% average discount on spot instances vs on-demand"
    note        = "Actual savings may vary based on instance availability and pricing"
  }
}

output "instance_types_configured" {
  description = "List of instance types configured for spot instances"
  value       = var.spot_instance_types
}

output "gpu_instance_types_configured" {
  description = "List of GPU instance types configured (if enabled)"
  value       = var.enable_gpu_instances ? var.gpu_instance_types : []
}

# Monitoring outputs
output "monitoring_configuration" {
  description = "Monitoring configuration for spot instances"
  value = {
    cloudwatch_agent_enabled = true
    log_group               = aws_cloudwatch_log_group.spot_instances.name
    detailed_monitoring     = var.enable_detailed_monitoring
    interruption_handler    = var.enable_interruption_handler
  }
}

# Scaling configuration outputs
output "scaling_configuration" {
  description = "Auto scaling configuration for spot instances"
  value = var.enable_spot_asg ? {
    min_size         = var.spot_min_size
    max_size         = var.spot_max_size
    desired_capacity = var.spot_desired_capacity
    on_demand_base   = var.on_demand_base_capacity
    on_demand_pct    = var.on_demand_percentage
    spot_max_price   = var.spot_max_price
  } : null
}

# Network configuration outputs
output "network_configuration" {
  description = "Network configuration for spot instances"
  value = {
    vpc_id                    = var.vpc_id
    subnet_ids               = var.private_subnet_ids
    security_group_id        = aws_security_group.spot_instances.id
    enhanced_networking      = var.enable_enhanced_networking
    placement_group_enabled  = var.enable_placement_group
    placement_group_strategy = var.placement_group_strategy
  }
}

