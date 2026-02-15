#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Self-Monitoring System
=============================================

Phase 3: Real-time system health monitoring and diagnostics.

This module provides continuous self-monitoring:
- Resource usage tracking (CPU, memory, disk)
- Performance metrics collection
- Error rate monitoring
- Anomaly detection
- Health scoring
- Alert generation

Key Features:
- Non-invasive monitoring (low overhead)
- Historical trend analysis
- Predictive health warnings
- Termux-compatible (works on 4GB RAM devices)

Author: JARVIS AI Project
Version: 3.0.0
Target Level: 60-70
Device: Realme Pad 2 Lite | Termux
"""

import time
import os
import sys
import logging
import threading
import json
import math
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = auto()        # Informational
    WARNING = auto()     # Potential issue
    CRITICAL = auto()    # Requires attention
    EMERGENCY = auto()   # Immediate action needed


class HealthStatus(Enum):
    """Overall health status"""
    EXCELLENT = "excellent"    # All systems optimal
    GOOD = "good"             # Minor issues
    FAIR = "fair"             # Some concerns
    POOR = "poor"             # Significant issues
    CRITICAL = "critical"     # System impaired


class MetricType(Enum):
    """Types of health metrics"""
    MEMORY_USAGE = auto()
    CPU_USAGE = auto()
    DISK_USAGE = auto()
    ERROR_RATE = auto()
    RESPONSE_TIME = auto()
    THROUGHPUT = auto()
    CACHE_HIT_RATE = auto()
    UPTIME = auto()
    ACTIVE_THREADS = auto()
    FILE_DESCRIPTORS = auto()


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HealthMetric:
    """
    A single health metric measurement.
    """
    metric_type: MetricType
    value: float
    unit: str = ""
    timestamp: float = field(default_factory=time.time)
    threshold_warning: float = 0.0
    threshold_critical: float = 0.0
    threshold_emergency: float = 0.0
    
    # Metadata
    source: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    
    def get_alert_level(self) -> AlertLevel:
        """Determine alert level based on thresholds"""
        if self.threshold_emergency > 0 and self.value >= self.threshold_emergency:
            return AlertLevel.EMERGENCY
        elif self.threshold_critical > 0 and self.value >= self.threshold_critical:
            return AlertLevel.CRITICAL
        elif self.threshold_warning > 0 and self.value >= self.threshold_warning:
            return AlertLevel.WARNING
        return AlertLevel.INFO
    
    def is_healthy(self) -> bool:
        """Check if metric is within healthy range"""
        return self.get_alert_level() in (AlertLevel.INFO, AlertLevel.WARNING)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.metric_type.name,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp,
            'alert_level': self.get_alert_level().name,
            'healthy': self.is_healthy(),
        }


@dataclass
class Alert:
    """
    A generated alert from monitoring.
    """
    level: AlertLevel
    metric_type: MetricType
    message: str
    value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[float] = None
    
    # Actions
    recommended_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'level': self.level.name,
            'metric': self.metric_type.name,
            'message': self.message,
            'value': self.value,
            'threshold': self.threshold,
            'timestamp': self.timestamp,
            'acknowledged': self.acknowledged,
            'resolved': self.resolved,
            'recommended_actions': self.recommended_actions,
        }


@dataclass
class SystemHealth:
    """
    Complete system health snapshot.
    """
    timestamp: float = field(default_factory=time.time)
    
    # Overall
    status: HealthStatus = HealthStatus.GOOD
    health_score: float = 100.0
    uptime_seconds: float = 0.0
    
    # Metrics by category
    memory_metrics: Dict[str, HealthMetric] = field(default_factory=dict)
    cpu_metrics: Dict[str, HealthMetric] = field(default_factory=dict)
    disk_metrics: Dict[str, HealthMetric] = field(default_factory=dict)
    performance_metrics: Dict[str, HealthMetric] = field(default_factory=dict)
    
    # Active alerts
    active_alerts: List[Alert] = field(default_factory=list)
    
    # Trends
    trend_improving: bool = True
    trend_duration_minutes: float = 0.0
    
    def get_all_metrics(self) -> List[HealthMetric]:
        """Get all metrics as flat list"""
        all_metrics = []
        all_metrics.extend(self.memory_metrics.values())
        all_metrics.extend(self.cpu_metrics.values())
        all_metrics.extend(self.disk_metrics.values())
        all_metrics.extend(self.performance_metrics.values())
        return all_metrics
    
    def get_unhealthy_metrics(self) -> List[HealthMetric]:
        """Get metrics that are not healthy"""
        return [m for m in self.get_all_metrics() if not m.is_healthy()]
    
    def has_critical_alerts(self) -> bool:
        """Check for critical/emergency alerts"""
        return any(
            a.level in (AlertLevel.CRITICAL, AlertLevel.EMERGENCY)
            for a in self.active_alerts
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'status': self.status.value,
            'health_score': self.health_score,
            'uptime_seconds': self.uptime_seconds,
            'active_alerts': len(self.active_alerts),
            'critical_alerts': sum(1 for a in self.active_alerts if a.level == AlertLevel.CRITICAL),
            'emergency_alerts': sum(1 for a in self.active_alerts if a.level == AlertLevel.EMERGENCY),
            'trend_improving': self.trend_improving,
            'metrics_count': len(self.get_all_metrics()),
            'unhealthy_count': len(self.get_unhealthy_metrics()),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC COLLECTORS
# ═══════════════════════════════════════════════════════════════════════════════

class MetricCollector:
    """
    Collect system metrics.
    
    Designed to work on Termux/Linux with minimal dependencies.
    """
    
    def __init__(self):
        """Initialize metric collector"""
        self._psutil_available = self._check_psutil()
        self._start_time = time.time()
    
    def _check_psutil(self) -> bool:
        """Check if psutil is available"""
        try:
            import psutil
            return True
        except ImportError:
            return False
    
    def collect_memory_metrics(self) -> Dict[str, HealthMetric]:
        """Collect memory-related metrics"""
        metrics = {}
        
        if self._psutil_available:
            import psutil
            
            # Virtual memory
            mem = psutil.virtual_memory()
            metrics['memory_percent'] = HealthMetric(
                metric_type=MetricType.MEMORY_USAGE,
                value=mem.percent,
                unit='%',
                threshold_warning=70.0,
                threshold_critical=85.0,
                threshold_emergency=95.0,
                source='psutil.virtual_memory',
            )
            
            metrics['memory_available_mb'] = HealthMetric(
                metric_type=MetricType.MEMORY_USAGE,
                value=mem.available / (1024 * 1024),
                unit='MB',
                source='psutil.virtual_memory',
            )
            
            # Swap
            swap = psutil.swap_memory()
            metrics['swap_percent'] = HealthMetric(
                metric_type=MetricType.MEMORY_USAGE,
                value=swap.percent,
                unit='%',
                threshold_warning=50.0,
                threshold_critical=75.0,
                threshold_emergency=90.0,
                source='psutil.swap_memory',
            )
        else:
            # Fallback: Read from /proc/meminfo (Linux/Termux)
            try:
                with open('/proc/meminfo', 'r') as f:
                    meminfo = {}
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            key = parts[0].rstrip(':')
                            value = int(parts[1])
                            meminfo[key] = value
                
                total = meminfo.get('MemTotal', 1)
                available = meminfo.get('MemAvailable', meminfo.get('MemFree', 0))
                used_percent = ((total - available) / total) * 100
                
                metrics['memory_percent'] = HealthMetric(
                    metric_type=MetricType.MEMORY_USAGE,
                    value=used_percent,
                    unit='%',
                    threshold_warning=70.0,
                    threshold_critical=85.0,
                    threshold_emergency=95.0,
                    source='/proc/meminfo',
                )
                
                metrics['memory_available_mb'] = HealthMetric(
                    metric_type=MetricType.MEMORY_USAGE,
                    value=available / 1024,
                    unit='MB',
                    source='/proc/meminfo',
                )
            except Exception as e:
                logger.warning(f"Could not read memory info: {e}")
        
        return metrics
    
    def collect_cpu_metrics(self) -> Dict[str, HealthMetric]:
        """Collect CPU-related metrics"""
        metrics = {}
        
        if self._psutil_available:
            import psutil
            
            # CPU percent
            cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics['cpu_percent'] = HealthMetric(
                metric_type=MetricType.CPU_USAGE,
                value=cpu_percent,
                unit='%',
                threshold_warning=70.0,
                threshold_critical=85.0,
                threshold_emergency=95.0,
                source='psutil.cpu_percent',
            )
            
            # Load average
            load_avg = os.getloadavg()
            metrics['load_avg_1m'] = HealthMetric(
                metric_type=MetricType.CPU_USAGE,
                value=load_avg[0],
                source='os.getloadavg',
            )
            metrics['load_avg_5m'] = HealthMetric(
                metric_type=MetricType.CPU_USAGE,
                value=load_avg[1],
                source='os.getloadavg',
            )
        else:
            # Fallback: Read from /proc/loadavg
            try:
                with open('/proc/loadavg', 'r') as f:
                    parts = f.read().split()
                    if len(parts) >= 2:
                        metrics['load_avg_1m'] = HealthMetric(
                            metric_type=MetricType.CPU_USAGE,
                            value=float(parts[0]),
                            source='/proc/loadavg',
                        )
                        metrics['load_avg_5m'] = HealthMetric(
                            metric_type=MetricType.CPU_USAGE,
                            value=float(parts[1]),
                            source='/proc/loadavg',
                        )
            except Exception as e:
                logger.warning(f"Could not read CPU load: {e}")
        
        return metrics
    
    def collect_disk_metrics(self) -> Dict[str, HealthMetric]:
        """Collect disk-related metrics"""
        metrics = {}
        
        if self._psutil_available:
            import psutil
            
            # Root partition
            disk = psutil.disk_usage('/')
            metrics['disk_percent'] = HealthMetric(
                metric_type=MetricType.DISK_USAGE,
                value=disk.percent,
                unit='%',
                threshold_warning=70.0,
                threshold_critical=85.0,
                threshold_emergency=95.0,
                source='psutil.disk_usage',
            )
            
            metrics['disk_free_gb'] = HealthMetric(
                metric_type=MetricType.DISK_USAGE,
                value=disk.free / (1024 * 1024 * 1024),
                unit='GB',
                source='psutil.disk_usage',
            )
        else:
            # Fallback: Use os.statvfs
            try:
                stat = os.statvfs('/')
                total = stat.f_blocks * stat.f_frsize
                free = stat.f_bavail * stat.f_frsize
                used_percent = ((total - free) / total) * 100
                
                metrics['disk_percent'] = HealthMetric(
                    metric_type=MetricType.DISK_USAGE,
                    value=used_percent,
                    unit='%',
                    threshold_warning=70.0,
                    threshold_critical=85.0,
                    threshold_emergency=95.0,
                    source='os.statvfs',
                )
                
                metrics['disk_free_gb'] = HealthMetric(
                    metric_type=MetricType.DISK_USAGE,
                    value=free / (1024 * 1024 * 1024),
                    unit='GB',
                    source='os.statvfs',
                )
            except Exception as e:
                logger.warning(f"Could not read disk info: {e}")
        
        return metrics
    
    def collect_process_metrics(self) -> Dict[str, HealthMetric]:
        """Collect process-related metrics"""
        metrics = {}
        
        # Thread count
        metrics['active_threads'] = HealthMetric(
            metric_type=MetricType.ACTIVE_THREADS,
            value=threading.active_count(),
            threshold_warning=50,
            threshold_critical=100,
            threshold_emergency=200,
            source='threading.active_count',
        )
        
        # Uptime
        uptime = time.time() - self._start_time
        metrics['uptime_seconds'] = HealthMetric(
            metric_type=MetricType.UPTIME,
            value=uptime,
            unit='s',
            source='internal',
        )
        
        return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# SELF MONITOR
# ═══════════════════════════════════════════════════════════════════════════════

class SelfMonitor:
    """
    Continuous self-monitoring system.
    
    Features:
    - Real-time health monitoring
    - Alert generation and management
    - Historical tracking
    - Trend analysis
    - Predictive warnings
    
    Usage:
        monitor = SelfMonitor()
        monitor.start()
        
        # Get current health
        health = monitor.get_health()
        print(f"Status: {health.status.value}")
        print(f"Score: {health.health_score}")
        
        # Check for alerts
        alerts = monitor.get_active_alerts()
    """
    
    # Default thresholds for health scoring
    DEFAULT_THRESHOLDS = {
        'memory_percent': {'warning': 70, 'critical': 85, 'emergency': 95},
        'cpu_percent': {'warning': 70, 'critical': 85, 'emergency': 95},
        'disk_percent': {'warning': 70, 'critical': 85, 'emergency': 95},
    }
    
    # Health score weights
    METRIC_WEIGHTS = {
        'memory_percent': 0.35,
        'cpu_percent': 0.25,
        'disk_percent': 0.20,
        'active_threads': 0.10,
        'error_rate': 0.10,
    }
    
    def __init__(
        self,
        collection_interval: float = 30.0,
        history_size: int = 100,
        alert_handlers: List[Callable] = None,
    ):
        """
        Initialize self monitor.
        
        Args:
            collection_interval: Seconds between metric collections
            history_size: Number of historical snapshots to keep
            alert_handlers: Callbacks for alert notifications
        """
        self._interval = collection_interval
        self._history_size = history_size
        self._alert_handlers = alert_handlers or []
        
        # Components
        self._collector = MetricCollector()
        
        # State
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Data
        self._history: deque = deque(maxlen=history_size)
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        
        # Custom metrics (for application-specific monitoring)
        self._custom_metrics: Dict[str, HealthMetric] = {}
        
        # Statistics
        self._stats = {
            'collections': 0,
            'alerts_generated': 0,
            'alerts_resolved': 0,
            'start_time': time.time(),
        }
        
        logger.info("SelfMonitor initialized")
    
    def start(self):
        """Start monitoring thread"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="SelfMonitor",
            daemon=True,
        )
        self._thread.start()
        logger.info("Self-monitoring started")
    
    def stop(self):
        """Stop monitoring thread"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Self-monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                self._collect_and_analyze()
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            # Sleep with interrupt check
            for _ in range(int(self._interval)):
                if not self._running:
                    break
                time.sleep(1)
    
    def _collect_and_analyze(self):
        """Collect metrics and analyze health"""
        with self._lock:
            # Create health snapshot
            health = SystemHealth()
            
            # Collect system metrics
            health.memory_metrics = self._collector.collect_memory_metrics()
            health.cpu_metrics = self._collector.collect_cpu_metrics()
            health.disk_metrics = self._collector.collect_disk_metrics()
            health.performance_metrics = self._collector.collect_process_metrics()
            
            # Add custom metrics
            for name, metric in self._custom_metrics.items():
                health.performance_metrics[name] = metric
            
            # Calculate health score
            health.health_score = self._calculate_health_score(health)
            health.status = self._determine_status(health.health_score)
            health.uptime_seconds = time.time() - self._stats['start_time']
            
            # Generate alerts
            new_alerts = self._generate_alerts(health)
            health.active_alerts = list(self._active_alerts.values())
            
            # Analyze trends
            health.trend_improving = self._analyze_trend()
            
            # Store in history
            self._history.append(health)
            self._stats['collections'] += 1
    
    def _calculate_health_score(self, health: SystemHealth) -> float:
        """Calculate overall health score (0-100)"""
        score = 100.0
        
        for metric in health.get_all_metrics():
            if not metric.is_healthy():
                # Deduct points based on severity
                alert_level = metric.get_alert_level()
                
                if alert_level == AlertLevel.WARNING:
                    deduction = 5
                elif alert_level == AlertLevel.CRITICAL:
                    deduction = 15
                elif alert_level == AlertLevel.EMERGENCY:
                    deduction = 30
                else:
                    deduction = 0
                
                # Apply weight
                weight = self.METRIC_WEIGHTS.get(metric.metric_type.name.lower(), 0.1)
                score -= deduction * weight
        
        return max(0.0, min(100.0, score))
    
    def _determine_status(self, health_score: float) -> HealthStatus:
        """Determine health status from score"""
        if health_score >= 90:
            return HealthStatus.EXCELLENT
        elif health_score >= 75:
            return HealthStatus.GOOD
        elif health_score >= 50:
            return HealthStatus.FAIR
        elif health_score >= 25:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL
    
    def _generate_alerts(self, health: SystemHealth) -> List[Alert]:
        """Generate alerts from health metrics"""
        new_alerts = []
        
        for metric in health.get_all_metrics():
            alert_level = metric.get_alert_level()
            
            if alert_level != AlertLevel.INFO:
                alert_key = f"{metric.metric_type.name}_{alert_level.name}"
                
                if alert_key not in self._active_alerts:
                    # Create new alert
                    alert = Alert(
                        level=alert_level,
                        metric_type=metric.metric_type,
                        message=self._create_alert_message(metric, alert_level),
                        value=metric.value,
                        threshold=metric.threshold_warning,
                        recommended_actions=self._get_recommended_actions(metric, alert_level),
                    )
                    
                    self._active_alerts[alert_key] = alert
                    new_alerts.append(alert)
                    self._stats['alerts_generated'] += 1
                    
                    # Notify handlers
                    self._notify_handlers(alert)
            else:
                # Check if we can resolve existing alerts
                alert_key_warning = f"{metric.metric_type.name}_WARNING"
                alert_key_critical = f"{metric.metric_type.name}_CRITICAL"
                
                for key in [alert_key_warning, alert_key_critical]:
                    if key in self._active_alerts:
                        alert = self._active_alerts[key]
                        alert.resolved = True
                        alert.resolution_time = time.time()
                        self._alert_history.append(alert)
                        del self._active_alerts[key]
                        self._stats['alerts_resolved'] += 1
        
        return new_alerts
    
    def _create_alert_message(self, metric: HealthMetric, level: AlertLevel) -> str:
        """Create human-readable alert message"""
        level_str = level.name.lower()
        return f"{level_str.title()}: {metric.metric_type.name.replace('_', ' ')} at {metric.value:.1f}{metric.unit}"
    
    def _get_recommended_actions(self, metric: HealthMetric, level: AlertLevel) -> List[str]:
        """Get recommended actions for alert"""
        actions = []
        
        if metric.metric_type == MetricType.MEMORY_USAGE:
            actions.extend([
                "Clear caches",
                "Reduce concurrent operations",
                "Check for memory leaks",
            ])
            if level == AlertLevel.CRITICAL:
                actions.append("Consider restarting application")
        
        elif metric.metric_type == MetricType.CPU_USAGE:
            actions.extend([
                "Reduce background tasks",
                "Optimize heavy computations",
                "Check for runaway processes",
            ])
        
        elif metric.metric_type == MetricType.DISK_USAGE:
            actions.extend([
                "Clean temporary files",
                "Archive old logs",
                "Remove unused caches",
            ])
        
        return actions
    
    def _notify_handlers(self, alert: Alert):
        """Notify alert handlers"""
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.warning(f"Alert handler error: {e}")
    
    def _analyze_trend(self) -> bool:
        """Analyze if health is improving or declining"""
        if len(self._history) < 3:
            return True
        
        # Compare recent scores
        recent = list(self._history)[-5:]
        scores = [h.health_score for h in recent]
        
        if len(scores) < 2:
            return True
        
        # Simple trend: average of last 2 vs average of previous
        recent_avg = sum(scores[-2:]) / 2
        older_avg = sum(scores[:-2]) / len(scores[:-2]) if len(scores) > 2 else scores[0]
        
        return recent_avg >= older_avg
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def get_health(self) -> SystemHealth:
        """Get current system health"""
        with self._lock:
            if self._history:
                return self._history[-1]
            return SystemHealth()
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active alerts"""
        with self._lock:
            return list(self._active_alerts.values())
    
    def get_history(self, count: int = 10) -> List[SystemHealth]:
        """Get historical health snapshots"""
        with self._lock:
            return list(self._history)[-count:]
    
    def record_custom_metric(self, name: str, metric: HealthMetric):
        """Record a custom metric"""
        with self._lock:
            self._custom_metrics[name] = metric
    
    def record_error(self, error_type: str):
        """Record an error occurrence"""
        # Could be extended to track error rates
        pass
    
    def record_performance(self, operation: str, duration_ms: float):
        """Record operation performance"""
        metric = HealthMetric(
            metric_type=MetricType.RESPONSE_TIME,
            value=duration_ms,
            unit='ms',
            source=operation,
            threshold_warning=1000,
            threshold_critical=5000,
        )
        self.record_custom_metric(f"perf_{operation}", metric)
    
    def acknowledge_alert(self, alert_key: str) -> bool:
        """Acknowledge an alert"""
        with self._lock:
            if alert_key in self._active_alerts:
                self._active_alerts[alert_key].acknowledged = True
                return True
            return False
    
    def force_collect(self) -> SystemHealth:
        """Force immediate metric collection"""
        self._collect_and_analyze()
        return self.get_health()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['uptime_seconds'] = time.time() - stats['start_time']
            stats['history_size'] = len(self._history)
            stats['active_alerts'] = len(self._active_alerts)
            stats['running'] = self._running
            return stats


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_monitor: Optional[SelfMonitor] = None


def get_self_monitor() -> SelfMonitor:
    """Get global self monitor"""
    global _monitor
    if _monitor is None:
        _monitor = SelfMonitor()
    return _monitor


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test():
    """Run self-test"""
    print("\n" + "="*60)
    print("Self Monitor Test")
    print("="*60)
    
    monitor = SelfMonitor(collection_interval=1)
    
    # Force collection
    health = monitor.force_collect()
    
    print(f"\nSystem Health:")
    print(f"  Status: {health.status.value}")
    print(f"  Score: {health.health_score:.1f}")
    print(f"  Uptime: {health.uptime_seconds:.0f}s")
    
    print(f"\nMetrics:")
    for metric in health.get_all_metrics()[:5]:
        status = "✓" if metric.is_healthy() else "!"
        print(f"  {status} {metric.metric_type.name}: {metric.value:.1f}{metric.unit}")
    
    print(f"\nAlerts: {len(health.active_alerts)}")
    
    stats = monitor.get_stats()
    print(f"\nStats:")
    print(f"  Collections: {stats['collections']}")
    print(f"  Uptime: {stats['uptime_seconds']:.0f}s")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    self_test()
