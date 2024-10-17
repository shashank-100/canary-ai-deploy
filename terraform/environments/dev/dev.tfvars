# Development environment configuration

aws_region   = "us-west-2"
project_name = "ai-model-serving"
environment  = "dev"

vpc_cidr = "10.0.0.0/16"

cluster_version = "1.28"

node_groups = {
  cpu_workers = {
    instance_types = ["t3.medium","t3.large"]
    capacity_type  = "ON_DEMAND"
    min_size      = 1
    max_size      = 4
    desired_size  = 2
    disk_size     = 50
    ami_type      = "AL2_x86_64"
    labels = {
      role = "cpu-worker"
    }
    taints = []
  }
  gpu_workers = {
    instance_types = ["g4dn.xlarge"]
    capacity_type  = "SPOT"
    min_size      = 0
    max_size      = 2
    desired_size  = 0
    disk_size     = 100
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
db_instance_class    = "db.t3.micro"
db_allocated_storage = 20

# Redis configuration
redis_node_type = "cache.t3.micro"

