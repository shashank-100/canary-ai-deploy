# Variables for Spot Instances Module

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
}

variable "vpc_id" {
  description = "VPC ID where spot instances will be created"
  type        = string
}

variable "private_subnet_ids" {
  description = "List of private subnet IDs for spot instances"
  type        = list(string)
}

variable "cluster_security_group_id" {
  description = "Security group ID of the EKS cluster"
  type        = string
}

variable "kubernetes_version" {
  description = "Kubernetes version for the worker nodes"
  type        = string
  default     = "1.27"
}

variable "region" {
  description = "AWS region"
  type        = string
}

variable "key_name" {
  description = "EC2 Key Pair name for SSH access"
  type        = string
  default     = ""
}

# Spot Fleet Configuration
variable "enable_spot_fleet" {
  description = "Enable spot fleet for batch processing"
  type        = bool
  default     = true
}

variable "spot_target_capacity" {
  description = "Target capacity for spot fleet"
  type        = number
  default     = 10
}

# Auto Scaling Group Configuration
variable "enable_spot_asg" {
  description = "Enable auto scaling group with spot instances"
  type        = bool
  default     = false
}

variable "spot_min_size" {
  description = "Minimum number of spot instances in ASG"
  type        = number
  default     = 0
}

variable "spot_max_size" {
  description = "Maximum number of spot instances in ASG"
  type        = number
  default     = 20
}

variable "spot_desired_capacity" {
  description = "Desired number of spot instances in ASG"
  type        = number
  default     = 2
}

variable "on_demand_base_capacity" {
  description = "Base number of on-demand instances"
  type        = number
  default     = 0
}

variable "on_demand_percentage" {
  description = "Percentage of on-demand instances above base capacity"
  type        = number
  default     = 0
}

variable "spot_max_price" {
  description = "Maximum price for spot instances"
  type        = string
  default     = ""
}

# Instance Configuration
variable "spot_instance_types" {
  description = "List of instance types for spot instances"
  type = list(object({
    instance_type = string
    max_price     = string
    weight        = number
  }))
  default = [
    {
      instance_type = "m5.large"
      max_price     = "0.10"
      weight        = 1
    },
    {
      instance_type = "m5.xlarge"
      max_price     = "0.20"
      weight        = 2
    },
    {
      instance_type = "m5.2xlarge"
      max_price     = "0.40"
      weight        = 4
    },
    {
      instance_type = "c5.large"
      max_price     = "0.09"
      weight        = 1
    },
    {
      instance_type = "c5.xlarge"
      max_price     = "0.18"
      weight        = 2
    },
    {
      instance_type = "c5.2xlarge"
      max_price     = "0.36"
      weight        = 4
    }
  ]
}

variable "root_volume_size" {
  description = "Size of the root EBS volume in GB"
  type        = number
  default     = 50
}

# GPU Instance Configuration
variable "enable_gpu_instances" {
  description = "Enable GPU instances for ML workloads"
  type        = bool
  default     = false
}

variable "gpu_instance_types" {
  description = "List of GPU instance types for spot instances"
  type = list(object({
    instance_type = string
    max_price     = string
    weight        = number
  }))
  default = [
    {
      instance_type = "g4dn.xlarge"
      max_price     = "0.50"
      weight        = 1
    },
    {
      instance_type = "g4dn.2xlarge"
      max_price     = "1.00"
      weight        = 2
    },
    {
      instance_type = "g4dn.4xlarge"
      max_price     = "2.00"
      weight        = 4
    }
  ]
}

# Interruption Handling
variable "enable_interruption_handler" {
  description = "Enable AWS Node Termination Handler for spot interruptions"
  type        = bool
  default     = true
}

# Monitoring and Logging
variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 7
}

variable "enable_detailed_monitoring" {
  description = "Enable detailed CloudWatch monitoring"
  type        = bool
  default     = false
}

# Tagging
variable "common_tags" {
  description = "Common tags to apply to all resources"
  type        = map(string)
  default = {
    Environment = "dev"
    Project     = "ai-model-serving"
    ManagedBy   = "terraform"
  }
}

# Cost Optimization
variable "enable_scheduled_scaling" {
  description = "Enable scheduled scaling for cost optimization"
  type        = bool
  default     = true
}

variable "scale_down_schedule" {
  description = "Cron expression for scaling down (UTC)"
  type        = string
  default     = "0 18 * * MON-FRI"  # 6 PM weekdays
}

variable "scale_up_schedule" {
  description = "Cron expression for scaling up (UTC)"
  type        = string
  default     = "0 8 * * MON-FRI"   # 8 AM weekdays
}

variable "weekend_capacity" {
  description = "Capacity during weekends"
  type        = number
  default     = 0
}

# Batch Processing Configuration
variable "batch_queue_name" {
  description = "Name of the batch processing queue"
  type        = string
  default     = "ai-model-batch-queue"
}

variable "batch_compute_environment_name" {
  description = "Name of the batch compute environment"
  type        = string
  default     = "ai-model-batch-compute"
}

variable "enable_batch_service" {
  description = "Enable AWS Batch service integration"
  type        = bool
  default     = false
}

# Networking
variable "enable_enhanced_networking" {
  description = "Enable enhanced networking (SR-IOV)"
  type        = bool
  default     = true
}

variable "enable_placement_group" {
  description = "Enable placement group for better network performance"
  type        = bool
  default     = false
}

variable "placement_group_strategy" {
  description = "Placement group strategy"
  type        = string
  default     = "cluster"
  validation {
    condition     = contains(["cluster", "partition", "spread"], var.placement_group_strategy)
    error_message = "Placement group strategy must be cluster, partition, or spread."
  }
}

