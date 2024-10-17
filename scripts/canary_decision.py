#!/usr/bin/env python3
"""
Canary Decision Script for AI Model Serving Platform
Makes promote/rollback decisions based on canary metrics
"""
import argparse
import json
import logging
import os
from typing import Dict, Any
import httpx
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CanaryDecisionMaker:
    """Makes decisions about canary deployments"""
    
    def __init__(self, model_name: str, prometheus_url: str = "http://prometheus:9090"):
        self.model_name = model_name
        self.prometheus_url = prometheus_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Decision thresholds
        self.max_error_rate = 0.01        # 1% max error rate
        self.max_latency_increase = 0.5   # 50% max latency increase
        self.min_throughput_ratio = 0.8   # 80% min throughput ratio
        self.max_resource_usage = 0.9     # 90% max resource usage
        
        # Weights for scoring
        self.weights = {
            "error_rate": 0.4,
            "latency": 0.3,
            "throughput": 0.2,
            "resources": 0.1
        }
    
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
                logger.error(f"Prometheus query failed: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to query Prometheus: {e}")
            return {}
    
    async def get_canary_metrics(self) -> Dict[str, float]:
        """Get current canary metrics"""
        metrics = {}
        
        # Error rate
        error_query = f'''
        rate(model_requests_total{{model_name="{self.model_name}", version="v2", status=~"5.."}}[10m]) /
        rate(model_requests_total{{model_name="{self.model_name}", version="v2"}}[10m])
        '''
        
        # Latency P95
        latency_query = f'''
        histogram_quantile(0.95, 
          rate(model_inference_duration_seconds_bucket{{model_name="{self.model_name}", version="v2"}}[10m])
        )
        '''
        
        # Throughput
        throughput_query = f'''
        rate(model_requests_total{{model_name="{self.model_name}", version="v2"}}[10m])
        '''
        
        # CPU usage
        cpu_query = f'''
        rate(container_cpu_usage_seconds_total{{pod=~"{self.model_name}-model-canary-.*"}}[10m])
        '''
        
        # Memory usage
        memory_query = f'''
        container_memory_usage_bytes{{pod=~"{self.model_name}-model-canary-.*"}} /
        container_spec_memory_limit_bytes{{pod=~"{self.model_name}-model-canary-.*"}}
        '''
        
        # Execute queries
        queries = {
            "error_rate": error_query,
            "latency_p95": latency_query,
            "throughput": throughput_query,
            "cpu_usage": cpu_query,
            "memory_usage": memory_query
        }
        
        for metric_name, query in queries.items():
            result = await self.query_prometheus(query)
            if result.get("status") == "success" and result.get("data", {}).get("result"):
                values = result["data"]["result"]
                if values:
                    metrics[metric_name] = float(values[0]["value"][1])
                else:
                    metrics[metric_name] = 0.0
            else:
                metrics[metric_name] = 0.0
        
        return metrics
    
    async def get_production_metrics(self) -> Dict[str, float]:
        """Get current production metrics"""
        metrics = {}
        
        # Error rate
        error_query = f'''
        rate(model_requests_total{{model_name="{self.model_name}", version="v1", status=~"5.."}}[10m]) /
        rate(model_requests_total{{model_name="{self.model_name}", version="v1"}}[10m])
        '''
        
        # Latency P95
        latency_query = f'''
        histogram_quantile(0.95, 
          rate(model_inference_duration_seconds_bucket{{model_name="{self.model_name}", version="v1"}}[10m])
        )
        '''
        
        # Throughput
        throughput_query = f'''
        rate(model_requests_total{{model_name="{self.model_name}", version="v1"}}[10m])
        '''
        
        # Execute queries
        queries = {
            "error_rate": error_query,
            "latency_p95": latency_query,
            "throughput": throughput_query
        }
        
        for metric_name, query in queries.items():
            result = await self.query_prometheus(query)
            if result.get("status") == "success" and result.get("data", {}).get("result"):
                values = result["data"]["result"]
                if values:
                    metrics[metric_name] = float(values[0]["value"][1])
                else:
                    metrics[metric_name] = 0.0
            else:
                metrics[metric_name] = 0.0
        
        return metrics
    
    def evaluate_error_rate(self, canary_metrics: Dict[str, float], 
                           production_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate error rate criteria"""
        canary_error_rate = canary_metrics.get("error_rate", 0)
        production_error_rate = production_metrics.get("error_rate", 0)
        
        # Check absolute threshold
        absolute_ok = canary_error_rate <= self.max_error_rate
        
        # Check relative to production (allow 2x increase max)
        relative_ok = canary_error_rate <= production_error_rate * 2
        
        # Score (0-1, higher is better)
        if canary_error_rate == 0:
            score = 1.0
        else:
            score = max(0, 1 - (canary_error_rate / self.max_error_rate))
        
        return {
            "metric": "error_rate",
            "canary_value": canary_error_rate,
            "production_value": production_error_rate,
            "absolute_ok": absolute_ok,
            "relative_ok": relative_ok,
            "overall_ok": absolute_ok and relative_ok,
            "score": score,
            "weight": self.weights["error_rate"],
            "weighted_score": score * self.weights["error_rate"]
        }
    
    def evaluate_latency(self, canary_metrics: Dict[str, float], 
                        production_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate latency criteria"""
        canary_latency = canary_metrics.get("latency_p95", 0)
        production_latency = production_metrics.get("latency_p95", 0)
        
        # Check if latency increase is acceptable
        if production_latency > 0:
            latency_increase = (canary_latency - production_latency) / production_latency
            relative_ok = latency_increase <= self.max_latency_increase
        else:
            latency_increase = 0
            relative_ok = True
        
        # Score based on latency increase
        if latency_increase <= 0:
            score = 1.0  # Latency improved
        elif latency_increase <= self.max_latency_increase:
            score = 1 - (latency_increase / self.max_latency_increase)
        else:
            score = 0.0
        
        return {
            "metric": "latency",
            "canary_value": canary_latency,
            "production_value": production_latency,
            "latency_increase": latency_increase,
            "relative_ok": relative_ok,
            "overall_ok": relative_ok,
            "score": score,
            "weight": self.weights["latency"],
            "weighted_score": score * self.weights["latency"]
        }
    
    def evaluate_throughput(self, canary_metrics: Dict[str, float], 
                           production_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate throughput criteria"""
        canary_throughput = canary_metrics.get("throughput", 0)
        production_throughput = production_metrics.get("throughput", 0)
        
        # Check if throughput is acceptable
        if production_throughput > 0:
            throughput_ratio = canary_throughput / production_throughput
            relative_ok = throughput_ratio >= self.min_throughput_ratio
        else:
            throughput_ratio = 1.0
            relative_ok = True
        
        # Score based on throughput ratio
        if throughput_ratio >= 1.0:
            score = 1.0  # Throughput improved
        elif throughput_ratio >= self.min_throughput_ratio:
            score = throughput_ratio / self.min_throughput_ratio
        else:
            score = 0.0
        
        return {
            "metric": "throughput",
            "canary_value": canary_throughput,
            "production_value": production_throughput,
            "throughput_ratio": throughput_ratio,
            "relative_ok": relative_ok,
            "overall_ok": relative_ok,
            "score": score,
            "weight": self.weights["throughput"],
            "weighted_score": score * self.weights["throughput"]
        }
    
    def evaluate_resources(self, canary_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate resource usage criteria"""
        cpu_usage = canary_metrics.get("cpu_usage", 0)
        memory_usage = canary_metrics.get("memory_usage", 0)
        
        # Check resource thresholds
        cpu_ok = cpu_usage <= self.max_resource_usage
        memory_ok = memory_usage <= self.max_resource_usage
        overall_ok = cpu_ok and memory_ok
        
        # Score based on resource usage
        max_usage = max(cpu_usage, memory_usage)
        if max_usage <= self.max_resource_usage:
            score = 1 - (max_usage / self.max_resource_usage)
        else:
            score = 0.0
        
        return {
            "metric": "resources",
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "cpu_ok": cpu_ok,
            "memory_ok": memory_ok,
            "overall_ok": overall_ok,
            "score": score,
            "weight": self.weights["resources"],
            "weighted_score": score * self.weights["resources"]
        }
    
    async def make_decision(self) -> Dict[str, Any]:
        """Make promote/rollback decision"""
        logger.info(f"Making canary decision for {self.model_name}")
        
        # Get metrics
        canary_metrics = await self.get_canary_metrics()
        production_metrics = await self.get_production_metrics()
        
        logger.info(f"Canary metrics: {canary_metrics}")
        logger.info(f"Production metrics: {production_metrics}")
        
        # Evaluate each criterion
        error_eval = self.evaluate_error_rate(canary_metrics, production_metrics)
        latency_eval = self.evaluate_latency(canary_metrics, production_metrics)
        throughput_eval = self.evaluate_throughput(canary_metrics, production_metrics)
        resources_eval = self.evaluate_resources(canary_metrics)
        
        evaluations = [error_eval, latency_eval, throughput_eval, resources_eval]
        
        # Calculate overall score
        total_weighted_score = sum(eval["weighted_score"] for eval in evaluations)
        
        # Check if all critical criteria pass
        all_criteria_pass = all(eval["overall_ok"] for eval in evaluations)
        
        # Decision logic
        if all_criteria_pass and total_weighted_score >= 0.8:
            decision = "promote"
            reason = "All criteria passed with good scores"
        elif all_criteria_pass and total_weighted_score >= 0.6:
            decision = "promote"
            reason = "All criteria passed with acceptable scores"
        else:
            decision = "rollback"
            failed_criteria = [eval["metric"] for eval in evaluations if not eval["overall_ok"]]
            reason = f"Failed criteria: {failed_criteria}, Score: {total_weighted_score:.3f}"
        
        # Additional safety checks
        if canary_metrics.get("error_rate", 0) > 0.05:  # 5% error rate is always a rollback
            decision = "rollback"
            reason = f"Critical error rate: {canary_metrics['error_rate']:.3f}"
        
        result = {
            "decision": decision,
            "reason": reason,
            "overall_score": total_weighted_score,
            "all_criteria_pass": all_criteria_pass,
            "canary_metrics": canary_metrics,
            "production_metrics": production_metrics,
            "evaluations": evaluations,
            "thresholds": {
                "max_error_rate": self.max_error_rate,
                "max_latency_increase": self.max_latency_increase,
                "min_throughput_ratio": self.min_throughput_ratio,
                "max_resource_usage": self.max_resource_usage
            }
        }
        
        return result
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.client.aclose()


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Canary deployment decision maker")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--prometheus-url", default="http://prometheus:9090", help="Prometheus URL")
    parser.add_argument("--output", help="Output file for decision details (JSON)")
    parser.add_argument("--config", help="Configuration file for thresholds (JSON)")
    
    args = parser.parse_args()
    
    # Create decision maker
    decision_maker = CanaryDecisionMaker(args.model, args.prometheus_url)
    
    # Load custom configuration if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
            
        # Update thresholds
        if "max_error_rate" in config:
            decision_maker.max_error_rate = config["max_error_rate"]
        if "max_latency_increase" in config:
            decision_maker.max_latency_increase = config["max_latency_increase"]
        if "min_throughput_ratio" in config:
            decision_maker.min_throughput_ratio = config["min_throughput_ratio"]
        if "max_resource_usage" in config:
            decision_maker.max_resource_usage = config["max_resource_usage"]
        if "weights" in config:
            decision_maker.weights.update(config["weights"])
    
    try:
        # Make decision
        result = await decision_maker.make_decision()
        
        # Print decision
        decision = result["decision"]
        reason = result["reason"]
        score = result["overall_score"]
        
        logger.info(f"Decision: {decision.upper()}")
        logger.info(f"Reason: {reason}")
        logger.info(f"Overall Score: {score:.3f}")
        
        # Print evaluation details
        logger.info("Evaluation Details:")
        for eval in result["evaluations"]:
            metric = eval["metric"]
            ok = eval["overall_ok"]
            score = eval["score"]
            status = "✅" if ok else "❌"
            logger.info(f"  {status} {metric}: score={score:.3f}, ok={ok}")
        
        # Save detailed results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Decision details saved to {args.output}")
        
        # Output decision for shell scripts
        print(decision)
        
        # Exit with appropriate code
        if decision == "promote":
            exit(0)
        else:
            exit(1)
            
    except Exception as e:
        logger.error(f"Decision making failed: {e}")
        print("rollback")  # Default to rollback on error
        exit(1)
    finally:
        await decision_maker.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

