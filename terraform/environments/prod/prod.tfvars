# Production environment configuration

aws_region   = "us-west-2"
project_name = "ai-model-serving"
environment  = "prod"

vpc_cidr = "10.0.0.0/16"

cluster_version = "1.28"

node_groups = {
  cpu_workers = {
    instance_types = ["m5.xlarge", "m5.2xlarge"]
    capacity_type  = "ON_DEMAND"
    min_size      = 2
    max_size      = 10
    desired_size  = 5
    disk_size     = 100
    ami_type      = "AL2_x86_64"
    labels = {
      role = "cpu-worker"
    }
    taints = []
  }
  gpu_workers = {
    instance_types = ["g4dn.2xlarge", "g4dn.4xlarge"]
    capacity_type  = "SPOT"
    min_size      = 1
    max_size      = 4
    desired_size  = 2
    disk_size     = 200
    ami_type      = "AL2_x86_64_GPU"
    labels = {
      role = "gpu-worker"
    }
    taints = [
      {
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }
    ]
  }
}

# Database configuration
db_instance_class    = "db.m5.large"
db_allocated_storage = 100

# Redis configuration
redis_node_type = "cache.m5.large" 