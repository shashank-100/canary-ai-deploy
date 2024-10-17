#!/usr/bin/env python3
"""
Canary Monitoring Script for AI Model Serving Platform
Monitors canary deployment metrics and alerts on issues
"""
import asyncio
import argparse
import time
import json
import logging
from typing import Dict, Any, List
import httpx
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CanaryMonitor:
    """Monitors canary deployment metrics"""
    
    def __init__(self, model_name: str, prometheus_url: str = "http://prometheus:9090"):
        self.model_name = model_name
        self.prometheus_url = prometheus_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Metric thresholds
        self.error_rate_threshold = 0.01  # 1%
        self.latency_threshold = 2.0      # 2 seconds
        self.cpu_threshold = 0.8          # 80%
        self.memory_threshold = 0.8       # 80%
    
    async def query_prometheus(self, query: str) -> Dict[str, Any]:
        """Query Prometheus for metrics"""
        try:
            response = await self.client.get(
                f"{self.prometheus_url}/api/v1/query",
                params={"query": query}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Prometheus query failed: {response.status_code} {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to query Prometheus: {e}")
            return {}
    
    async def query_prometheus_range(self, query: str, start: str, end: str, step: str = "30s") -> Dict[str, Any]:
        """Query Prometheus for range data"""
        try:
            response = await self.client.get(
                f"{self.prometheus_url}/api/v1/query_range",
                params={
                    "query": query,
                    "start": start,
                    "end": end,
                    "step": step
                }
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Prometheus range query failed: {response.status_code} {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to query Prometheus range: {e}")
            return {}
    
    async def get_error_rate(self, version: str = "v2") -> float:
        """Get error rate for canary deployment"""
        query = f'''
        rate(model_requests_total{{model_name="{self.model_name}", version="{version}", status=~"5.."}}[5m]) /
        rate(model_requests_total{{model_name="{self.model_name}", version="{version}"}}[5m])
        '''
        
        result = await self.query_prometheus(query)
        
        if result.get("status") == "success" and result.get("data", {}).get("result"):
            values = result["data"]["result"]
            if values:
                return float(values[0]["value"][1])
        
        return 0.0
    
    async def get_latency_percentile(self, percentile: int = 95, version: str = "v2") -> float:
        """Get latency percentile for canary deployment"""
        query = f'''
        histogram_quantile(0.{percentile:02d}, 
          rate(model_inference_duration_seconds_bucket{{model_name="{self.model_name}", version="{version}"}}[5m])
        )
        '''
        
        result = await self.query_prometheus(query)
        
        if result.get("status") == "success" and result.get("data", {}).get("result"):
            values = result["data"]["result"]
            if values:
                return float(values[0]["value"][1])
        
        return 0.0
    
    async def get_throughput(self, version: str = "v2") -> float:
        """Get throughput for canary deployment"""
        query = f'''
        rate(model_requests_total{{model_name="{self.model_name}", version="{version}"}}[5m])
        '''
        
        result = await self.query_prometheus(query)
        
        if result.get("status") == "success" and result.get("data", {}).get("result"):
            values = result["data"]["result"]
            if values:
                return float(values[0]["value"][1])
        
        return 0.0
    
    async def get_resource_usage(self, version: str = "v2") -> Dict[str, float]:
        """Get resource usage for canary deployment"""
        # CPU usage
        cpu_query = f'''
        rate(container_cpu_usage_seconds_total{{pod=~"{self.model_name}-model-canary-.*"}}[5m])
        '''
        
        # Memory usage
        memory_query = f'''
        container_memory_usage_bytes{{pod=~"{self.model_name}-model-canary-.*"}} /
        container_spec_memory_limit_bytes{{pod=~"{self.model_name}-model-canary-.*"}}
        '''
        
        cpu_result = await self.query_prometheus(cpu_query)
        memory_result = await self.query_prometheus(memory_query)
        
        cpu_usage = 0.0
        memory_usage = 0.0
        
        if cpu_result.get("status") == "success" and cpu_result.get("data", {}).get("result"):
            values = cpu_result["data"]["result"]
            if values:
                cpu_usage = float(values[0]["value"][1])
        
        if memory_result.get("status") == "success" and memory_result.get("data", {}).get("result"):
            values = memory_result["data"]["result"]
            if values:
                memory_usage = float(values[0]["value"][1])
        
        return {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage
        }
    
    async def compare_with_production(self) -> Dict[str, Any]:
        """Compare canary metrics with production"""
        # Get metrics for both versions
        canary_error_rate = await self.get_error_rate("v2")
        production_error_rate = await self.get_error_rate("v1")
        
        canary_latency = await self.get_latency_percentile(95, "v2")
        production_latency = await self.get_latency_percentile(95, "v1")
        
        canary_throughput = await self.get_throughput("v2")
        production_throughput = await self.get_throughput("v1")
        
        # Calculate differences
        error_rate_diff = canary_error_rate - production_error_rate
        latency_diff = canary_latency - production_latency
        throughput_diff = canary_throughput - production_throughput
        
        return {
            "canary": {
                "error_rate": canary_error_rate,
                "latency_p95": canary_latency,
                "throughput": canary_throughput
            },
            "production": {
                "error_rate": production_error_rate,
                "latency_p95": production_latency,
                "throughput": production_throughput
            },
            "differences": {
                "error_rate_diff": error_rate_diff,
                "latency_diff": latency_diff,
                "throughput_diff": throughput_diff
            }
        }
    
    async def check_health_metrics(self) -> Dict[str, Any]:
        """Check overall health metrics"""
        error_rate = await self.get_error_rate("v2")
        latency_p95 = await self.get_latency_percentile(95, "v2")
        latency_p99 = await self.get_latency_percentile(99, "v2")
        throughput = await self.get_throughput("v2")
        resources = await self.get_resource_usage("v2")
        
        # Check thresholds
        issues = []
        
        if error_rate > self.error_rate_threshold:
            issues.append(f"High error rate: {error_rate:.3f} > {self.error_rate_threshold}")
        
        if latency_p95 > self.latency_threshold:
            issues.append(f"High P95 latency: {latency_p95:.3f}s > {self.latency_threshold}s")
        
        if resources["cpu_usage"] > self.cpu_threshold:
            issues.append(f"High CPU usage: {resources['cpu_usage']:.3f} > {self.cpu_threshold}")
        
        if resources["memory_usage"] > self.memory_threshold:
            issues.append(f"High memory usage: {resources['memory_usage']:.3f} > {self.memory_threshold}")
        
        return {
            "healthy": len(issues) == 0,
            "metrics": {
                "error_rate": error_rate,
                "latency_p95": latency_p95,
                "latency_p99": latency_p99,
                "throughput": throughput,
                "cpu_usage": resources["cpu_usage"],
                "memory_usage": resources["memory_usage"]
            },
            "issues": issues,
            "thresholds": {
                "error_rate": self.error_rate_threshold,
                "latency": self.latency_threshold,
                "cpu": self.cpu_threshold,
                "memory": self.memory_threshold
            }
        }
    
    async def monitor_for_duration(self, duration_seconds: int, check_interval: int = 30) -> Dict[str, Any]:
        """Monitor canary for specified duration"""
        logger.info(f"Starting canary monitoring for {duration_seconds} seconds")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        health_checks = []
        comparison_checks = []
        
        while time.time() < end_time:
            current_time = time.time()
            elapsed = current_time - start_time
            remaining = end_time - current_time
            
            logger.info(f"Monitoring progress: {elapsed:.0f}s elapsed, {remaining:.0f}s remaining")
            
            # Check health metrics
            health = await self.check_health_metrics()
            health["timestamp"] = datetime.utcnow().isoformat()
            health_checks.append(health)
            
            # Compare with production
            comparison = await self.compare_with_production()
            comparison["timestamp"] = datetime.utcnow().isoformat()
            comparison_checks.append(comparison)
            
            # Log current status
            if health["healthy"]:
                logger.info("✅ Canary health check passed")
            else:
                logger.warning(f"⚠️ Canary health issues: {health['issues']}")
            
            # Log comparison
            diff = comparison["differences"]
            logger.info(f"📊 Comparison - Error rate diff: {diff['error_rate_diff']:.4f}, "
                       f"Latency diff: {diff['latency_diff']:.3f}s")
            
            # Wait for next check
            if remaining > check_interval:
                await asyncio.sleep(check_interval)
            else:
                break
        
        # Calculate summary
        healthy_checks = sum(1 for check in health_checks if check["healthy"])
        health_rate = healthy_checks / len(health_checks) if health_checks else 0
        
        # Get final metrics
        final_health = health_checks[-1] if health_checks else {}
        final_comparison = comparison_checks[-1] if comparison_checks else {}
        
        # Determine overall status
        overall_healthy = (
            health_rate >= 0.9 and  # 90% of checks must be healthy
            final_health.get("healthy", False)
        )
        
        return {
            "overall_healthy": overall_healthy,
            "monitoring_duration": duration_seconds,
            "total_checks": len(health_checks),
            "healthy_checks": healthy_checks,
            "health_rate": health_rate,
            "final_metrics": final_health.get("metrics", {}),
            "final_comparison": final_comparison,
            "all_health_checks": health_checks,
            "all_comparisons": comparison_checks
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.client.aclose()


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Canary deployment monitor")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--duration", type=int, default=300, help="Monitoring duration in seconds")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    parser.add_argument("--prometheus-url", default="http://prometheus:9090", help="Prometheus URL")
    parser.add_argument("--error-threshold", type=float, default=0.01, help="Error rate threshold")
    parser.add_argument("--latency-threshold", type=float, default=2.0, help="Latency threshold in seconds")
    parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    # Create monitor
    monitor = CanaryMonitor(args.model, args.prometheus_url)
    monitor.error_rate_threshold = args.error_threshold
    monitor.latency_threshold = args.latency_threshold
    
    try:
        # Run monitoring
        logger.info(f"Monitoring {args.model} canary deployment")
        results = await monitor.monitor_for_duration(args.duration, args.interval)
        
        # Print results
        logger.info("Monitoring Results:")
        logger.info(f"  Overall Healthy: {results['overall_healthy']}")
        logger.info(f"  Health Rate: {results['health_rate']:.2%}")
        logger.info(f"  Total Checks: {results['total_checks']}")
        logger.info(f"  Healthy Checks: {results['healthy_checks']}")
        
        if results.get('final_metrics'):
            metrics = results['final_metrics']
            logger.info("  Final Metrics:")
            logger.info(f"    Error Rate: {metrics.get('error_rate', 0):.4f}")
            logger.info(f"    P95 Latency: {metrics.get('latency_p95', 0):.3f}s")
            logger.info(f"    Throughput: {metrics.get('throughput', 0):.2f} req/s")
            logger.info(f"    CPU Usage: {metrics.get('cpu_usage', 0):.3f}")
            logger.info(f"    Memory Usage: {metrics.get('memory_usage', 0):.3f}")
        
        # Save results to file if specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        
        # Exit with appropriate code
        if results['overall_healthy']:
            logger.info("✅ Canary monitoring PASSED")
            exit(0)
        else:
            logger.error("❌ Canary monitoring FAILED")
            exit(1)
            
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        exit(1)
    finally:
        await monitor.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

