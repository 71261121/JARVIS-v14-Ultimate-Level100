#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Multi-Agent Orchestration Module
=======================================================

Phase 4: Advanced Multi-Agent Orchestration System (Level 70-80)

This module provides sophisticated multi-agent coordination and
distributed task execution capabilities for JARVIS.

Components:
-----------
- MultiAgentOrchestrator: Main orchestration system
- Agent Collaboration: Communication, consensus, and messaging
- Distributed Executor: Task scheduling and load balancing
- Workflow Manager: Complex workflow DAG management

Key Features:
-------------
- Agent lifecycle management (spawn, monitor, terminate)
- Dynamic agent specialization and role assignment
- Inter-agent negotiation protocols
- Hierarchical task decomposition
- Consensus building and conflict resolution
- Distributed decision making
- DAG-based workflow definition
- Dependency resolution and execution

Usage:
------
    from core.agents import (
        MultiAgentOrchestrator,
        AgentCollaborationManager,
        DistributedExecutor,
        WorkflowManager,
    )
    
    # Create orchestrator
    orchestrator = MultiAgentOrchestrator(kimi_client)
    orchestrator.start()
    
    # Spawn agents
    agent_id = orchestrator.spawn_agent(profile_name='executor')
    
    # Execute workflow
    workflow = workflow_manager.create_workflow("My Workflow")
    workflow_manager.add_step(workflow.id, "Step 1", task_func)
    instance = workflow_manager.execute(workflow.id)

Author: JARVIS AI Project
Version: 4.0.0
Target Level: 70-80
"""

# Multi-Agent Orchestrator
from .multi_agent_orchestrator import (
    # Enums
    AgentState,
    AgentRole,
    AgentCapability,
    TaskPriority,
    NegotiationState,
    ConsensusType,
    ConflictType,
    DecompositionStrategy,
    
    # Dataclasses
    AgentProfile,
    AgentMetrics,
    SubTask,
    TaskDecomposition,
    Proposal,
    Negotiation,
    Conflict,
    
    # Classes
    Agent,
    DefaultAgent,
    TaskDecomposer,
    NegotiationEngine,
    ConflictResolver,
    CapabilityMatcher,
    MultiAgentOrchestrator,
    
    # Functions
    get_orchestrator,
)

# Agent Collaboration
from .agent_collaboration import (
    # Enums
    MessageType,
    ConsensusState,
    VoteType,
    TeamRole,
    EventType,
    ChannelType,
    Priority,
    
    # Dataclasses
    Message,
    Vote,
    Proposal as CollaborationProposal,
    KnowledgeEntry,
    Team,
    Event,
    
    # Classes
    MessageRouter,
    ConsensusEngine,
    SharedKnowledgeBase,
    TeamManager,
    EventBus,
    AgentCollaborationManager,
    
    # Functions
    get_collaboration_manager,
)

# Distributed Executor
from .distributed_executor import (
    # Enums
    TaskStatus,
    TaskPriority as ExecutorTaskPriority,
    ScheduleStrategy,
    LoadBalanceStrategy,
    ExecutionMode,
    ResourceType,
    FailureType,
    
    # Dataclasses
    Task,
    WorkerInfo,
    ExecutionPlan,
    ExecutionResult,
    ResourceAllocation,
    
    # Classes
    TaskScheduler,
    LoadBalancer,
    FaultTolerance,
    ResultAggregator,
    DistributedExecutor,
    
    # Functions
    get_executor,
)

# Workflow Manager
from .workflow_manager import (
    # Enums
    WorkflowStatus,
    StepStatus,
    StepType,
    DependencyType,
    WorkflowPriority,
    BranchCondition,
    LoopType,
    
    # Dataclasses
    Dependency,
    StepResult,
    WorkflowStep,
    WorkflowDefinition,
    WorkflowInstance,
    
    # Classes
    DependencyResolver,
    WorkflowExecutor,
    WorkflowTemplateEngine,
    WorkflowStateManager,
    WorkflowManager,
    
    # Functions
    get_workflow_manager,
)


__all__ = [
    # Multi-Agent Orchestrator
    'AgentState',
    'AgentRole', 
    'AgentCapability',
    'TaskPriority',
    'NegotiationState',
    'ConsensusType',
    'ConflictType',
    'DecompositionStrategy',
    'AgentProfile',
    'AgentMetrics',
    'SubTask',
    'TaskDecomposition',
    'Proposal',
    'Negotiation',
    'Conflict',
    'Agent',
    'DefaultAgent',
    'TaskDecomposer',
    'NegotiationEngine',
    'ConflictResolver',
    'CapabilityMatcher',
    'MultiAgentOrchestrator',
    'get_orchestrator',
    
    # Agent Collaboration
    'MessageType',
    'ConsensusState',
    'VoteType',
    'TeamRole',
    'EventType',
    'ChannelType',
    'Priority',
    'Message',
    'Vote',
    'KnowledgeEntry',
    'Team',
    'Event',
    'MessageRouter',
    'ConsensusEngine',
    'SharedKnowledgeBase',
    'TeamManager',
    'EventBus',
    'AgentCollaborationManager',
    'get_collaboration_manager',
    
    # Distributed Executor
    'TaskStatus',
    'ExecutorTaskPriority',
    'ScheduleStrategy',
    'LoadBalanceStrategy',
    'ExecutionMode',
    'ResourceType',
    'FailureType',
    'Task',
    'WorkerInfo',
    'ExecutionPlan',
    'ExecutionResult',
    'ResourceAllocation',
    'TaskScheduler',
    'LoadBalancer',
    'FaultTolerance',
    'ResultAggregator',
    'DistributedExecutor',
    'get_executor',
    
    # Workflow Manager
    'WorkflowStatus',
    'StepStatus',
    'StepType',
    'DependencyType',
    'WorkflowPriority',
    'BranchCondition',
    'LoopType',
    'Dependency',
    'StepResult',
    'WorkflowStep',
    'WorkflowDefinition',
    'WorkflowInstance',
    'DependencyResolver',
    'WorkflowExecutor',
    'WorkflowTemplateEngine',
    'WorkflowStateManager',
    'WorkflowManager',
    'get_workflow_manager',
]


# Version info
__version__ = "4.0.0"
__phase__ = "Phase 4: Advanced Multi-Agent Orchestration"
__target_level__ = "Level 70-80"


def initialize_phase4(kimi_client=None):
    """
    Initialize all Phase 4 components.
    
    Args:
        kimi_client: Kimi K2.5 client for AI reasoning
        
    Returns:
        Dictionary with initialized components
    """
    # Initialize orchestrator
    orchestrator = get_orchestrator(kimi_client=kimi_client)
    orchestrator.start()
    
    # Initialize collaboration manager
    collaboration = get_collaboration_manager()
    
    # Initialize distributed executor
    executor = get_executor()
    executor.start()
    
    # Initialize workflow manager
    workflow_manager = get_workflow_manager()
    
    return {
        'orchestrator': orchestrator,
        'collaboration': collaboration,
        'executor': executor,
        'workflow_manager': workflow_manager,
    }


def get_phase4_status() -> dict:
    """
    Get status of all Phase 4 components.
    
    Returns:
        Dictionary with component statuses
    """
    try:
        orchestrator = get_orchestrator()
        executor = get_executor()
        workflow_manager = get_workflow_manager()
        
        return {
            'phase': __phase__,
            'version': __version__,
            'target_level': __target_level__,
            'orchestrator': orchestrator.get_stats(),
            'executor': executor.get_stats(),
            'workflow_manager': workflow_manager.get_stats(),
        }
    except Exception as e:
        return {
            'phase': __phase__,
            'version': __version__,
            'error': str(e),
        }
