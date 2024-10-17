# Spot Instances Module for Cost Optimization
# Provides cost-effective compute for batch inference jobs

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_ami" "eks_worker" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["amazon-eks-node-${var.kubernetes_version}-v*"]
  }
}

# Spot Fleet Request for batch processing nodes
resource "aws_spot_fleet_request" "batch_processing" {
  count = var.enable_spot_fleet ? 1 : 0

  iam_fleet_role                      = aws_iam_role.spot_fleet_role[0].arn
  allocation_strategy                 = "diversified"
  target_capacity                     = var.spot_target_capacity
  fleet_type                         = "maintain"
  terminate_instances_with_expiration = true
  wait_for_fulfillment               = false
  replace_unhealthy_instances        = true

  # Launch specifications for different instance types
  dynamic "launch_specification" {
    for_each = var.spot_instance_types
    content {
      image_id                    = data.aws_ami.eks_worker.id
      instance_type              = launch_specification.value.instance_type
      key_name                   = var.key_name
      vpc_security_group_ids     = [aws_security_group.spot_instances.id]
      subnet_id                  = element(var.private_subnet_ids, count.index % length(var.private_subnet_ids))
      iam_instance_profile       = aws_iam_instance_profile.spot_instance_profile.name
      availability_zone          = data.aws_availability_zones.available.names[launch_specification.key % length(data.aws_availability_zones.available.names)]
      spot_price                 = launch_specification.value.max_price
      weighted_capacity          = launch_specification.value.weight

      user_data = base64encode(templatefile("${path.module}/user_data.sh", {
        cluster_name     = var.cluster_name
        node_group_name  = "spot-batch-processing"
        kubernetes_version = var.kubernetes_version
        region          = var.region
      }))

      root_block_device {
        volume_type           = "gp3"
        volume_size           = var.root_volume_size
        delete_on_termination = true
        encrypted            = true
      }

      tags = merge(var.common_tags, {
        Name = "${var.cluster_name}-spot-batch-${launch_specification.value.instance_type}"
        "kubernetes.io/cluster/${var.cluster_name}" = "owned"
        "k8s.io/cluster-autoscaler/enabled" = "true"
        "k8s.io/cluster-autoscaler/${var.cluster_name}" = "owned"
        "k8s.io/cluster-autoscaler/node-template/label/node-type" = "spot-batch"
        "k8s.io/cluster-autoscaler/node-template/label/instance-type" = launch_specification.value.instance_type
      })
    }
  }

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-spot-fleet"
  })
}

# Auto Scaling Group for spot instances (alternative approach)
resource "aws_autoscaling_group" "spot_batch_processing" {
  count = var.enable_spot_asg ? 1 : 0

  name                = "${var.cluster_name}-spot-batch-asg"
  vpc_zone_identifier = var.private_subnet_ids
  target_group_arns   = []
  health_check_type   = "EC2"
  health_check_grace_period = 300

  min_size         = var.spot_min_size
  max_size         = var.spot_max_size
  desired_capacity = var.spot_desired_capacity

  # Mixed instances policy for cost optimization
  mixed_instances_policy {
    instances_distribution {
      on_demand_base_capacity                  = var.on_demand_base_capacity
      on_demand_percentage_above_base_capacity = var.on_demand_percentage
      spot_allocation_strategy                 = "capacity-optimized"
      spot_instance_pools                      = 3
      spot_max_price                          = var.spot_max_price
    }

    launch_template {
      launch_template_specification {
        launch_template_id = aws_launch_template.spot_batch_processing.id
        version           = "$Latest"
      }

      dynamic "override" {
        for_each = var.spot_instance_types
        content {
          instance_type     = override.value.instance_type
          weighted_capacity = override.value.weight
        }
      }
    }
  }

  # Instance refresh configuration
  instance_refresh {
    strategy = "Rolling"
    preferences {
      min_healthy_percentage = 50
      instance_warmup       = 300
    }
  }

  tag {
    key                 = "Name"
    value               = "${var.cluster_name}-spot-batch-node"
    propagate_at_launch = true
  }

  tag {
    key                 = "kubernetes.io/cluster/${var.cluster_name}"
    value               = "owned"
    propagate_at_launch = true
  }

  tag {
    key                 = "k8s.io/cluster-autoscaler/enabled"
    value               = "true"
    propagate_at_launch = true
  }

  tag {
    key                 = "k8s.io/cluster-autoscaler/${var.cluster_name}"
    value               = "owned"
    propagate_at_launch = true
  }

  tag {
    key                 = "k8s.io/cluster-autoscaler/node-template/label/node-type"
    value               = "spot-batch"
    propagate_at_launch = true
  }

  dynamic "tag" {
    for_each = var.common_tags
    content {
      key                 = tag.key
      value               = tag.value
      propagate_at_launch = true
    }
  }
}

# Launch template for spot instances
resource "aws_launch_template" "spot_batch_processing" {
  name_prefix   = "${var.cluster_name}-spot-batch-"
  image_id      = data.aws_ami.eks_worker.id
  key_name      = var.key_name
  vpc_security_group_ids = [aws_security_group.spot_instances.id]

  iam_instance_profile {
    name = aws_iam_instance_profile.spot_instance_profile.name
  }

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = var.root_volume_size
      volume_type           = "gp3"
      delete_on_termination = true
      encrypted            = true
    }
  }

  monitoring {
    enabled = true
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    cluster_name     = var.cluster_name
    node_group_name  = "spot-batch-processing"
    kubernetes_version = var.kubernetes_version
    region          = var.region
  }))

  tag_specifications {
    resource_type = "instance"
    tags = merge(var.common_tags, {
      Name = "${var.cluster_name}-spot-batch-node"
      "kubernetes.io/cluster/${var.cluster_name}" = "owned"
    })
  }

  tag_specifications {
    resource_type = "volume"
    tags = merge(var.common_tags, {
      Name = "${var.cluster_name}-spot-batch-volume"
    })
  }

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-spot-batch-template"
  })
}

# Security group for spot instances
resource "aws_security_group" "spot_instances" {
  name_prefix = "${var.cluster_name}-spot-batch-"
  vpc_id      = var.vpc_id

  # Allow all traffic from cluster security group
  ingress {
    from_port       = 0
    to_port         = 65535
    protocol        = "tcp"
    security_groups = [var.cluster_security_group_id]
  }

  # Allow all traffic to cluster security group
  egress {
    from_port       = 0
    to_port         = 65535
    protocol        = "tcp"
    security_groups = [var.cluster_security_group_id]
  }

  # Allow HTTPS outbound for package downloads
  egress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow HTTP outbound for package downloads
  egress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow DNS
  egress {
    from_port   = 53
    to_port     = 53
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-spot-batch-sg"
  })
}

# IAM role for spot fleet
resource "aws_iam_role" "spot_fleet_role" {
  count = var.enable_spot_fleet ? 1 : 0

  name = "${var.cluster_name}-spot-fleet-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "spotfleet.amazonaws.com"
        }
      }
    ]
  })

  tags = var.common_tags
}

# Attach spot fleet policy
resource "aws_iam_role_policy_attachment" "spot_fleet_policy" {
  count = var.enable_spot_fleet ? 1 : 0

  role       = aws_iam_role.spot_fleet_role[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2SpotFleetTaggingRole"
}

# IAM role for spot instances
resource "aws_iam_role" "spot_instance_role" {
  name = "${var.cluster_name}-spot-instance-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = var.common_tags
}

# Attach required policies to spot instance role
resource "aws_iam_role_policy_attachment" "spot_instance_worker_policy" {
  role       = aws_iam_role.spot_instance_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

resource "aws_iam_role_policy_attachment" "spot_instance_cni_policy" {
  role       = aws_iam_role.spot_instance_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

resource "aws_iam_role_policy_attachment" "spot_instance_registry_policy" {
  role       = aws_iam_role.spot_instance_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

# Instance profile for spot instances
resource "aws_iam_instance_profile" "spot_instance_profile" {
  name = "${var.cluster_name}-spot-instance-profile"
  role = aws_iam_role.spot_instance_role.name

  tags = var.common_tags
}

# CloudWatch Log Group for spot instances
resource "aws_cloudwatch_log_group" "spot_instances" {
  name              = "/aws/eks/${var.cluster_name}/spot-batch"
  retention_in_days = var.log_retention_days

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-spot-batch-logs"
  })
}

# Spot Instance Interruption Handler (using AWS Node Termination Handler)
resource "aws_sqs_queue" "spot_interruption" {
  count = var.enable_interruption_handler ? 1 : 0

  name                      = "${var.cluster_name}-spot-interruption"
  message_retention_seconds = 300

  tags = merge(var.common_tags, {
    Name = "${var.cluster_name}-spot-interruption-queue"
  })
}

# EventBridge rule for spot interruption
resource "aws_cloudwatch_event_rule" "spot_interruption" {
  count = var.enable_interruption_handler ? 1 : 0

  name        = "${var.cluster_name}-spot-interruption"
  description = "Capture spot instance interruption warnings"

  event_pattern = jsonencode({
    source      = ["aws.ec2"]
    detail-type = ["EC2 Spot Instance Interruption Warning"]
  })

  tags = var.common_tags
}

# EventBridge target for spot interruption
resource "aws_cloudwatch_event_target" "spot_interruption" {
  count = var.enable_interruption_handler ? 1 : 0

  rule      = aws_cloudwatch_event_rule.spot_interruption[0].name
  target_id = "SpotInterruptionTarget"
  arn       = aws_sqs_queue.spot_interruption[0].arn
}

