#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JARVIS v14 Ultimate - Agent Collaboration Module
=================================================

Phase 4: Advanced Agent Communication and Consensus (Level 70-80)

This module provides sophisticated inter-agent collaboration:
- Communication protocols and message passing
- Consensus building algorithms (Byzantine, Raft, Paxos-style)
- Shared memory and knowledge exchange
- Agent groups and teams
- Voting and decision making
- Broadcast and multicast messaging
- Event-driven collaboration
- Secure communication channels

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│                   Agent Collaboration Layer                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Message    │  │  Consensus  │  │   Shared    │  Core        │
│  │   Router    │  │   Engine    │  │   Memory    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Voting    │  │   Team      │  │   Event     │  Management  │
│  │   System    │  │   Manager   │  │   Bus       │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Broadcast  │  │  Knowledge  │  │   Secure    │  Channels    │
│  │   Manager   │  │   Exchange  │  │  Channels   │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────────────────────────────────────────────┘

Author: JARVIS AI Project
Version: 4.0.0
Target Level: 70-80
"""

import time
import json
import logging
import threading
import uuid
import hashlib
import math
import random
from typing import Dict, Any, Optional, List, Set, Callable, Tuple, Union, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
from collections import deque, defaultdict
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue, PriorityQueue
import weakref

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class MessageType(Enum):
    """Types of messages between agents"""
    # Task related
    TASK_REQUEST = auto()
    TASK_RESPONSE = auto()
    TASK_STATUS = auto()
    TASK_RESULT = auto()
    
    # Communication
    QUERY = auto()
    RESPONSE = auto()
    NOTIFICATION = auto()
    ACKNOWLEDGMENT = auto()
    
    # Collaboration
    COLLABORATION_REQUEST = auto()
    COLLABORATION_ACCEPT = auto()
    COLLABORATION_REJECT = auto()
    COLLABORATION_UPDATE = auto()
    
    # Knowledge
    KNOWLEDGE_SHARE = auto()
    KNOWLEDGE_REQUEST = auto()
    KNOWLEDGE_UPDATE = auto()
    
    # Consensus
    VOTE_REQUEST = auto()
    VOTE_RESPONSE = auto()
    LEADER_HEARTBEAT = auto()
    LEADER_ELECTION = auto()
    
    # System
    HEARTBEAT = auto()
    STATUS_UPDATE = auto()
    ERROR_REPORT = auto()
    SHUTDOWN = auto()


class ConsensusState(Enum):
    """States in consensus process"""
    IDLE = auto()
    PROPOSING = auto()
    VOTING = auto()
    COMMITTING = auto()
    COMMITTED = auto()
    ABORTED = auto()


class VoteType(Enum):
    """Types of votes"""
    YES = auto()
    NO = auto()
    ABSTAIN = auto()
    VETO = auto()


class TeamRole(Enum):
    """Roles within a team"""
    LEADER = auto()
    MEMBER = auto()
    SPECIALIST = auto()
    OBSERVER = auto()
    COORDINATOR = auto()


class EventType(Enum):
    """Types of events in the system"""
    AGENT_JOINED = auto()
    AGENT_LEFT = auto()
    TASK_STARTED = auto()
    TASK_COMPLETED = auto()
    TASK_FAILED = auto()
    CONSENSUS_REACHED = auto()
    CONSENSUS_FAILED = auto()
    KNOWLEDGE_UPDATED = auto()
    ERROR_OCCURRED = auto()
    THRESHOLD_EXCEEDED = auto()
    GOAL_ACHIEVED = auto()
    GOAL_MISSED = auto()


class ChannelType(Enum):
    """Types of communication channels"""
    DIRECT = auto()        # Point-to-point
    BROADCAST = auto()     # One-to-all
    MULTICAST = auto()     # One-to-many
    PUBLISH_SUBSCRIBE = auto()  # Pub-sub pattern
    REQUEST_REPLY = auto() # Request-reply pattern


class Priority(Enum):
    """Message priority levels"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Message:
    """
    A message between agents.
    
    Contains all information needed for communication.
    """
    # Identification
    id: str = field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:8]}")
    correlation_id: Optional[str] = None  # For request-response correlation
    
    # Routing
    sender_id: str = ""
    receiver_id: str = ""  # For direct messages
    channel: str = ""  # For channel-based messaging
    
    # Content
    message_type: MessageType = MessageType.NOTIFICATION
    priority: Priority = Priority.NORMAL
    content: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    ttl: float = 60.0  # Time-to-live in seconds
    requires_ack: bool = False
    ack_timeout: float = 5.0
    
    # Delivery tracking
    sent: bool = False
    delivered: bool = False
    acknowledged: bool = False
    delivery_attempts: int = 0
    max_attempts: int = 3
    
    # Compression/Encryption
    compressed: bool = False
    encrypted: bool = False
    
    @property
    def is_expired(self) -> bool:
        """Check if message has expired"""
        return time.time() - self.timestamp > self.ttl
    
    @property
    def is_broadcast(self) -> bool:
        """Check if message is a broadcast"""
        return self.receiver_id == "*" or self.channel.startswith("broadcast:")
    
    def create_response(
        self,
        response_type: MessageType,
        content: Dict[str, Any],
    ) -> 'Message':
        """Create a response to this message"""
        return Message(
            correlation_id=self.id,
            sender_id=self.receiver_id,
            receiver_id=self.sender_id,
            message_type=response_type,
            content=content,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'sender': self.sender_id,
            'receiver': self.receiver_id,
            'type': self.message_type.name,
            'priority': self.priority.name,
            'content': self.content,
            'timestamp': self.timestamp,
        }


@dataclass
class Vote:
    """
    A vote in a consensus process.
    """
    id: str = field(default_factory=lambda: f"vote_{uuid.uuid4().hex[:8]}")
    proposal_id: str = ""
    voter_id: str = ""
    vote_type: VoteType = VoteType.YES
    weight: float = 1.0
    reason: str = ""
    timestamp: float = field(default_factory=time.time)
    
    # Conditions
    conditions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'proposal_id': self.proposal_id,
            'voter_id': self.voter_id,
            'vote': self.vote_type.name,
            'weight': self.weight,
            'reason': self.reason,
        }


@dataclass
class Proposal:
    """
    A proposal for consensus.
    """
    id: str = field(default_factory=lambda: f"prop_{uuid.uuid4().hex[:8]}")
    proposer_id: str = ""
    title: str = ""
    description: str = ""
    
    # Content
    content: Dict[str, Any] = field(default_factory=dict)
    
    # Voting
    votes: Dict[str, Vote] = field(default_factory=dict)
    required_votes: int = 1
    required_weight: float = 1.0
    quorum: float = 0.5  # Minimum participation
    
    # Timing
    created_at: float = field(default_factory=time.time)
    voting_deadline: float = 300.0  # Seconds from creation
    state: ConsensusState = ConsensusState.IDLE
    
    # Outcome
    outcome: Optional[str] = None  # "approved", "rejected", "expired"
    decided_at: Optional[float] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if proposal has expired"""
        return time.time() - self.created_at > self.voting_deadline
    
    @property
    def total_votes(self) -> int:
        """Get total vote count"""
        return len(self.votes)
    
    @property
    def yes_votes(self) -> int:
        """Get yes vote count"""
        return sum(1 for v in self.votes.values() if v.vote_type == VoteType.YES)
    
    @property
    def no_votes(self) -> int:
        """Get no vote count"""
        return sum(1 for v in self.votes.values() if v.vote_type == VoteType.NO)
    
    @property
    def yes_weight(self) -> float:
        """Get total yes weight"""
        return sum(v.weight for v in self.votes.values() if v.vote_type == VoteType.YES)
    
    @property
    def total_weight(self) -> float:
        """Get total voting weight"""
        return sum(v.weight for v in self.votes.values())
    
    def add_vote(self, vote: Vote):
        """Add a vote to the proposal"""
        vote.proposal_id = self.id
        self.votes[vote.voter_id] = vote
    
    def check_consensus(self) -> Optional[str]:
        """Check if consensus has been reached"""
        if self.is_expired:
            self.state = ConsensusState.ABORTED
            self.outcome = "expired"
            return "expired"
        
        # Check for veto
        for vote in self.votes.values():
            if vote.vote_type == VoteType.VETO:
                self.state = ConsensusState.ABORTED
                self.outcome = "rejected"
                self.decided_at = time.time()
                return "rejected"
        
        # Check required votes
        if self.yes_votes >= self.required_votes:
            self.state = ConsensusState.COMMITTED
            self.outcome = "approved"
            self.decided_at = time.time()
            return "approved"
        
        # Check if impossible to reach consensus
        remaining_voters = 0  # Would need member count
        if self.yes_votes + remaining_voters < self.required_votes:
            self.state = ConsensusState.ABORTED
            self.outcome = "rejected"
            self.decided_at = time.time()
            return "rejected"
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'proposer': self.proposer_id,
            'title': self.title,
            'state': self.state.name,
            'outcome': self.outcome,
            'votes': {
                'total': self.total_votes,
                'yes': self.yes_votes,
                'no': self.no_votes,
            },
        }


@dataclass
class KnowledgeEntry:
    """
    An entry in the shared knowledge base.
    """
    id: str = field(default_factory=lambda: f"know_{uuid.uuid4().hex[:8]}")
    key: str = ""
    value: Any = None
    
    # Metadata
    source_agent: str = ""
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    
    # Versioning
    version: int = 1
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    # Access
    read_count: int = 0
    last_read_at: Optional[float] = None
    
    # Expiration
    ttl: Optional[float] = None  # Time-to-live in seconds
    
    # Confidence
    confidence: float = 1.0
    verified: bool = False
    verified_by: List[str] = field(default_factory=list)
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.updated_at > self.ttl
    
    def update(self, new_value: Any, source: str = ""):
        """Update the entry"""
        self.value = new_value
        self.version += 1
        self.updated_at = time.time()
        if source:
            self.source_agent = source
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'key': self.key,
            'value': self.value,
            'version': self.version,
            'source': self.source_agent,
            'confidence': self.confidence,
        }


@dataclass
class Team:
    """
    A team of agents working together.
    """
    id: str = field(default_factory=lambda: f"team_{uuid.uuid4().hex[:8]}")
    name: str = ""
    description: str = ""
    
    # Members
    members: Dict[str, TeamRole] = field(default_factory=dict)
    leader_id: Optional[str] = None
    max_members: int = 10
    
    # Purpose
    goal: str = ""
    capabilities_required: Set[str] = field(default_factory=set)
    
    # Communication
    channel_id: str = ""
    message_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    # State
    active: bool = True
    created_at: float = field(default_factory=time.time)
    
    # Performance
    tasks_completed: int = 0
    tasks_failed: int = 0
    
    @property
    def member_count(self) -> int:
        return len(self.members)
    
    @property
    def is_full(self) -> bool:
        return len(self.members) >= self.max_members
    
    def add_member(self, agent_id: str, role: TeamRole = TeamRole.MEMBER) -> bool:
        """Add a member to the team"""
        if self.is_full:
            return False
        
        self.members[agent_id] = role
        
        if role == TeamRole.LEADER:
            self.leader_id = agent_id
        
        return True
    
    def remove_member(self, agent_id: str) -> bool:
        """Remove a member from the team"""
        if agent_id not in self.members:
            return False
        
        del self.members[agent_id]
        
        if agent_id == self.leader_id:
            # Need to elect new leader
            self.leader_id = None
        
        return True
    
    def get_members_by_role(self, role: TeamRole) -> List[str]:
        """Get members with a specific role"""
        return [aid for aid, r in self.members.items() if r == role]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'members': len(self.members),
            'leader': self.leader_id,
            'active': self.active,
            'tasks_completed': self.tasks_completed,
        }


@dataclass
class Event:
    """
    An event in the system.
    """
    id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:8]}")
    event_type: EventType = EventType.NOTIFICATION
    source: str = ""
    
    # Content
    data: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    severity: str = "info"  # debug, info, warning, error, critical
    
    # Propagation
    propagated: bool = False
    subscribers_notified: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.event_type.name,
            'source': self.source,
            'message': self.message,
            'timestamp': self.timestamp,
            'severity': self.severity,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MESSAGE ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

class MessageRouter:
    """
    Routes messages between agents.
    
    Handles direct, broadcast, and multicast messaging.
    """
    
    def __init__(self, max_queue_size: int = 10000):
        """Initialize message router."""
        # Agent queues
        self._agent_queues: Dict[str, Queue] = {}
        
        # Channels
        self._channels: Dict[str, Set[str]] = {}  # channel -> subscribers
        
        # Message tracking
        self._pending_acks: Dict[str, Message] = {}
        self._message_history: deque = deque(maxlen=10000)
        
        # Statistics
        self._stats = {
            'total_routed': 0,
            'direct_messages': 0,
            'broadcasts': 0,
            'multicasts': 0,
            'failed_deliveries': 0,
            'avg_delivery_time': 0.0,
        }
        
        self._lock = threading.RLock()
        
        logger.info("MessageRouter initialized")
    
    def register_agent(self, agent_id: str, queue_size: int = 100) -> Queue:
        """Register an agent and get its message queue"""
        with self._lock:
            queue = Queue(maxsize=queue_size)
            self._agent_queues[agent_id] = queue
            logger.debug(f"Registered agent {agent_id} with message queue")
            return queue
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        with self._lock:
            if agent_id in self._agent_queues:
                del self._agent_queues[agent_id]
            
            # Remove from all channels
            for subscribers in self._channels.values():
                subscribers.discard(agent_id)
    
    def create_channel(self, channel_id: str) -> bool:
        """Create a new communication channel"""
        with self._lock:
            if channel_id in self._channels:
                return False
            
            self._channels[channel_id] = set()
            logger.debug(f"Created channel {channel_id}")
            return True
    
    def subscribe(self, agent_id: str, channel_id: str) -> bool:
        """Subscribe an agent to a channel"""
        with self._lock:
            if channel_id not in self._channels:
                self._channels[channel_id] = set()
            
            self._channels[channel_id].add(agent_id)
            logger.debug(f"Agent {agent_id} subscribed to {channel_id}")
            return True
    
    def unsubscribe(self, agent_id: str, channel_id: str) -> bool:
        """Unsubscribe an agent from a channel"""
        with self._lock:
            if channel_id not in self._channels:
                return False
            
            if agent_id in self._channels[channel_id]:
                self._channels[channel_id].discard(agent_id)
                return True
            return False
    
    def route(self, message: Message) -> int:
        """
        Route a message.
        
        Returns number of recipients.
        """
        with self._lock:
            if message.is_expired:
                return 0
            
            recipients = 0
            start_time = time.time()
            
            # Direct message
            if message.receiver_id and message.receiver_id != "*":
                recipients = self._deliver_direct(message)
                self._stats['direct_messages'] += 1
            
            # Broadcast
            elif message.receiver_id == "*":
                recipients = self._deliver_broadcast(message)
                self._stats['broadcasts'] += 1
            
            # Channel-based
            elif message.channel:
                recipients = self._deliver_to_channel(message)
                self._stats['multicasts'] += 1
            
            # Track
            message.sent = True
            self._message_history.append(message)
            self._stats['total_routed'] += 1
            
            # Track for ACK
            if message.requires_ack:
                self._pending_acks[message.id] = message
            
            # Update avg delivery time
            delivery_time = time.time() - start_time
            self._stats['avg_delivery_time'] = (
                self._stats['avg_delivery_time'] * 0.9 + delivery_time * 0.1
            )
            
            return recipients
    
    def _deliver_direct(self, message: Message) -> int:
        """Deliver message to single recipient"""
        queue = self._agent_queues.get(message.receiver_id)
        if not queue:
            self._stats['failed_deliveries'] += 1
            return 0
        
        try:
            queue.put_nowait(message)
            message.delivered = True
            return 1
        except:
            self._stats['failed_deliveries'] += 1
            return 0
    
    def _deliver_broadcast(self, message: Message) -> int:
        """Deliver message to all agents"""
        recipients = 0
        for agent_id, queue in self._agent_queues.items():
            if agent_id == message.sender_id:
                continue
            
            try:
                queue.put_nowait(message)
                recipients += 1
            except:
                pass
        
        message.delivered = True
        return recipients
    
    def _deliver_to_channel(self, message: Message) -> int:
        """Deliver message to channel subscribers"""
        channel = message.channel
        if channel.startswith("broadcast:"):
            channel = channel.split(":")[1]
        
        subscribers = self._channels.get(channel, set())
        recipients = 0
        
        for agent_id in subscribers:
            if agent_id == message.sender_id:
                continue
            
            queue = self._agent_queues.get(agent_id)
            if queue:
                try:
                    queue.put_nowait(message)
                    recipients += 1
                except:
                    pass
        
        message.delivered = True
        return recipients
    
    def acknowledge(self, message_id: str, ack_from: str) -> bool:
        """Acknowledge a message"""
        with self._lock:
            if message_id in self._pending_acks:
                message = self._pending_acks.pop(message_id)
                message.acknowledged = True
                return True
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['registered_agents'] = len(self._agent_queues)
            stats['active_channels'] = len(self._channels)
            stats['pending_acks'] = len(self._pending_acks)
            return stats


# ═══════════════════════════════════════════════════════════════════════════════
# CONSENSUS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class ConsensusEngine:
    """
    Implements consensus algorithms for distributed decision making.
    
    Supports multiple consensus types:
    - Simple majority voting
    - Weighted voting
    - Byzantine fault tolerance
    - Raft-style leader election
    """
    
    def __init__(self, node_id: str = None):
        """Initialize consensus engine."""
        self._node_id = node_id or f"node_{uuid.uuid4().hex[:8]}"
        
        # Proposals
        self._active_proposals: Dict[str, Proposal] = {}
        self._completed_proposals: deque = deque(maxlen=1000)
        
        # Voting state
        self._voting_rights: Dict[str, float] = {}  # agent_id -> weight
        self._voting_history: Dict[str, List[str]] = defaultdict(list)
        
        # Leader election (Raft-style)
        self._current_term: int = 0
        self._current_leader: Optional[str] = None
        self._last_heartbeat: float = 0.0
        self._election_timeout: float = 5.0
        
        # Byzantine fault tolerance
        self._byzantine_threshold: float = 0.33  # Can tolerate 1/3 faulty nodes
        
        # Statistics
        self._stats = {
            'proposals_created': 0,
            'proposals_approved': 0,
            'proposals_rejected': 0,
            'proposals_expired': 0,
            'total_votes_cast': 0,
            'leader_changes': 0,
        }
        
        self._lock = threading.RLock()
        
        logger.info(f"ConsensusEngine initialized for {self._node_id}")
    
    def set_voting_rights(self, agent_id: str, weight: float = 1.0):
        """Set voting rights for an agent"""
        with self._lock:
            self._voting_rights[agent_id] = weight
    
    def remove_voting_rights(self, agent_id: str):
        """Remove voting rights for an agent"""
        with self._lock:
            if agent_id in self._voting_rights:
                del self._voting_rights[agent_id]
    
    def create_proposal(
        self,
        title: str,
        description: str,
        content: Dict[str, Any],
        required_votes: int = None,
        required_weight: float = None,
        deadline: float = 300.0,
    ) -> Proposal:
        """
        Create a new proposal for voting.
        
        Args:
            title: Proposal title
            description: Detailed description
            content: Proposal content/payload
            required_votes: Minimum yes votes needed
            required_weight: Minimum yes weight needed
            deadline: Voting deadline in seconds
            
        Returns:
            Proposal object
        """
        with self._lock:
            # Calculate defaults
            if required_votes is None:
                required_votes = max(1, len(self._voting_rights) // 2 + 1)
            
            if required_weight is None:
                total_weight = sum(self._voting_rights.values())
                required_weight = total_weight / 2
            
            proposal = Proposal(
                proposer_id=self._node_id,
                title=title,
                description=description,
                content=content,
                required_votes=required_votes,
                required_weight=required_weight,
                voting_deadline=deadline,
                state=ConsensusState.PROPOSING,
            )
            
            self._active_proposals[proposal.id] = proposal
            self._stats['proposals_created'] += 1
            
            logger.info(f"Created proposal {proposal.id}: {title}")
            return proposal
    
    def cast_vote(
        self,
        proposal_id: str,
        voter_id: str,
        vote_type: VoteType,
        reason: str = "",
        conditions: List[str] = None,
    ) -> Optional[Vote]:
        """
        Cast a vote on a proposal.
        
        Args:
            proposal_id: Proposal to vote on
            voter_id: Agent casting vote
            vote_type: YES, NO, ABSTAIN, or VETO
            reason: Reason for vote
            conditions: Conditions for approval
            
        Returns:
            Vote object or None if invalid
        """
        with self._lock:
            proposal = self._active_proposals.get(proposal_id)
            if not proposal:
                return None
            
            if proposal.is_expired or proposal.outcome:
                return None
            
            # Check voting rights
            weight = self._voting_rights.get(voter_id, 1.0)
            
            vote = Vote(
                proposal_id=proposal_id,
                voter_id=voter_id,
                vote_type=vote_type,
                weight=weight,
                reason=reason,
                conditions=conditions or [],
            )
            
            proposal.add_vote(vote)
            self._voting_history[voter_id].append(proposal_id)
            self._stats['total_votes_cast'] += 1
            
            # Check if consensus reached
            outcome = proposal.check_consensus()
            
            if outcome == "approved":
                self._stats['proposals_approved'] += 1
                self._complete_proposal(proposal)
            elif outcome == "rejected":
                self._stats['proposals_rejected'] += 1
                self._complete_proposal(proposal)
            elif outcome == "expired":
                self._stats['proposals_expired'] += 1
                self._complete_proposal(proposal)
            
            logger.debug(
                f"Vote cast by {voter_id} on {proposal_id}: {vote_type.name}"
            )
            
            return vote
    
    def check_timeouts(self):
        """Check for expired proposals"""
        with self._lock:
            expired = []
            
            for proposal_id, proposal in self._active_proposals.items():
                if proposal.is_expired:
                    expired.append(proposal_id)
            
            for proposal_id in expired:
                proposal = self._active_proposals.pop(proposal_id)
                proposal.state = ConsensusState.ABORTED
                proposal.outcome = "expired"
                self._complete_proposal(proposal)
                self._stats['proposals_expired'] += 1
    
    def _complete_proposal(self, proposal: Proposal):
        """Move proposal to completed"""
        if proposal.id in self._active_proposals:
            del self._active_proposals[proposal.id]
        self._completed_proposals.append(proposal)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Raft-style Leader Election
    # ─────────────────────────────────────────────────────────────────────────
    
    def start_election(self) -> bool:
        """
        Start a leader election (Raft-style).
        
        Returns True if this node becomes leader.
        """
        with self._lock:
            self._current_term += 1
            
            # Request votes from all nodes
            votes_received = 1  # Vote for self
            total_voters = len(self._voting_rights)
            
            # In a real implementation, would send vote requests
            # and wait for responses
            
            # Simple simulation: check if we have majority
            if votes_received > total_voters / 2:
                self._current_leader = self._node_id
                self._last_heartbeat = time.time()
                self._stats['leader_changes'] += 1
                logger.info(f"Node {self._node_id} elected leader for term {self._current_term}")
                return True
            
            return False
    
    def send_heartbeat(self):
        """Send leader heartbeat"""
        with self._lock:
            if self._current_leader == self._node_id:
                self._last_heartbeat = time.time()
    
    def receive_heartbeat(self, leader_id: str, term: int):
        """Receive heartbeat from leader"""
        with self._lock:
            if term >= self._current_term:
                self._current_term = term
                self._current_leader = leader_id
                self._last_heartbeat = time.time()
    
    def check_leader_timeout(self) -> bool:
        """Check if leader has timed out"""
        with self._lock:
            if time.time() - self._last_heartbeat > self._election_timeout:
                return True
            return False
    
    # ─────────────────────────────────────────────────────────────────────────
    # Byzantine Fault Tolerance
    # ─────────────────────────────────────────────────────────────────────────
    
    def byzantine_vote(
        self,
        proposal_id: str,
        voter_id: str,
        vote_type: VoteType,
        round_num: int = 0,
    ) -> Dict[str, Any]:
        """
        Byzantine fault tolerant voting.
        
        Implements simplified PBFT-style consensus.
        """
        with self._lock:
            proposal = self._active_proposals.get(proposal_id)
            if not proposal:
                return {'success': False, 'error': 'Proposal not found'}
            
            # Phase 1: Pre-prepare
            # Phase 2: Prepare
            # Phase 3: Commit
            
            # Simplified: just check for supermajority
            total_nodes = len(self._voting_rights)
            required = math.ceil(total_nodes * (2/3 + 0.01))
            
            weight = self._voting_rights.get(voter_id, 1.0)
            
            vote = Vote(
                proposal_id=proposal_id,
                voter_id=voter_id,
                vote_type=vote_type,
                weight=weight,
            )
            
            proposal.add_vote(vote)
            
            # Check for Byzantine consensus
            if proposal.yes_votes >= required:
                proposal.state = ConsensusState.COMMITTED
                proposal.outcome = "approved"
                self._complete_proposal(proposal)
                self._stats['proposals_approved'] += 1
                return {'success': True, 'consensus': True}
            
            return {'success': True, 'consensus': False}
    
    def get_proposal(self, proposal_id: str) -> Optional[Proposal]:
        """Get proposal by ID"""
        return self._active_proposals.get(proposal_id)
    
    def get_active_proposals(self) -> List[Proposal]:
        """Get all active proposals"""
        with self._lock:
            return list(self._active_proposals.values())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get consensus statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['active_proposals'] = len(self._active_proposals)
            stats['current_leader'] = self._current_leader
            stats['current_term'] = self._current_term
            stats['voters'] = len(self._voting_rights)
            return stats


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════════

class SharedKnowledgeBase:
    """
    Shared knowledge base for agents.
    
    Provides a common knowledge repository with:
    - Key-value storage
    - Versioning
    - Expiration
    - Confidence tracking
    """
    
    def __init__(self, max_entries: int = 10000):
        """Initialize knowledge base."""
        self._entries: Dict[str, KnowledgeEntry] = {}
        self._categories: Dict[str, Set[str]] = defaultdict(set)
        self._tags: Dict[str, Set[str]] = defaultdict(set)
        
        # Index for fast lookup
        self._by_source: Dict[str, Set[str]] = defaultdict(set)
        
        # Statistics
        self._stats = {
            'total_entries': 0,
            'total_reads': 0,
            'total_writes': 0,
            'total_updates': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }
        
        self._lock = threading.RLock()
        
        logger.info("SharedKnowledgeBase initialized")
    
    def store(
        self,
        key: str,
        value: Any,
        source_agent: str = "",
        category: str = "general",
        tags: List[str] = None,
        ttl: float = None,
        confidence: float = 1.0,
    ) -> KnowledgeEntry:
        """
        Store a knowledge entry.
        
        Args:
            key: Unique key
            value: Value to store
            source_agent: Agent providing the knowledge
            category: Category for organization
            tags: Tags for search
            ttl: Time-to-live in seconds
            confidence: Confidence level (0-1)
            
        Returns:
            KnowledgeEntry
        """
        with self._lock:
            # Check if updating existing
            if key in self._entries:
                entry = self._entries[key]
                entry.update(value, source_agent)
                entry.confidence = confidence
                self._stats['total_updates'] += 1
            else:
                entry = KnowledgeEntry(
                    key=key,
                    value=value,
                    source_agent=source_agent,
                    category=category,
                    tags=tags or [],
                    ttl=ttl,
                    confidence=confidence,
                )
                self._entries[key] = entry
                self._stats['total_entries'] += 1
            
            # Update indices
            self._categories[category].add(key)
            
            for tag in (tags or []):
                self._tags[tag].add(key)
            
            if source_agent:
                self._by_source[source_agent].add(key)
            
            self._stats['total_writes'] += 1
            
            logger.debug(f"Stored knowledge: {key} from {source_agent}")
            return entry
    
    def retrieve(self, key: str) -> Optional[KnowledgeEntry]:
        """Retrieve a knowledge entry"""
        with self._lock:
            entry = self._entries.get(key)
            
            if entry is None:
                self._stats['cache_misses'] += 1
                return None
            
            if entry.is_expired:
                self._delete_entry(key)
                self._stats['cache_misses'] += 1
                return None
            
            entry.read_count += 1
            entry.last_read_at = time.time()
            self._stats['total_reads'] += 1
            self._stats['cache_hits'] += 1
            
            return entry
    
    def query(
        self,
        category: str = None,
        tags: List[str] = None,
        source: str = None,
        min_confidence: float = 0.0,
    ) -> List[KnowledgeEntry]:
        """
        Query knowledge entries.
        
        Args:
            category: Filter by category
            tags: Filter by tags (any match)
            source: Filter by source agent
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of matching entries
        """
        with self._lock:
            # Start with all keys
            keys = set(self._entries.keys())
            
            # Filter by category
            if category:
                keys &= self._categories.get(category, set())
            
            # Filter by tags
            if tags:
                tag_keys = set()
                for tag in tags:
                    tag_keys |= self._tags.get(tag, set())
                keys &= tag_keys
            
            # Filter by source
            if source:
                keys &= self._by_source.get(source, set())
            
            # Get entries and filter by confidence
            results = []
            for key in keys:
                entry = self._entries.get(key)
                if entry and not entry.is_expired and entry.confidence >= min_confidence:
                    results.append(entry)
            
            self._stats['total_reads'] += len(results)
            return results
    
    def verify(self, key: str, verifier: str) -> bool:
        """Mark an entry as verified"""
        with self._lock:
            entry = self._entries.get(key)
            if not entry:
                return False
            
            if verifier not in entry.verified_by:
                entry.verified_by.append(verifier)
                entry.verified = True
                entry.confidence = min(1.0, entry.confidence + 0.1)
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete a knowledge entry"""
        with self._lock:
            if key not in self._entries:
                return False
            
            self._delete_entry(key)
            return True
    
    def _delete_entry(self, key: str):
        """Internal delete without lock"""
        entry = self._entries.pop(key)
        
        # Remove from indices
        self._categories[entry.category].discard(key)
        
        for tag in entry.tags:
            self._tags[tag].discard(key)
        
        if entry.source_agent:
            self._by_source[entry.source_agent].discard(key)
        
        self._stats['total_entries'] -= 1
    
    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        with self._lock:
            expired = [
                key for key, entry in self._entries.items()
                if entry.is_expired
            ]
            
            for key in expired:
                self._delete_entry(key)
            
            return len(expired)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['categories'] = len(self._categories)
            stats['unique_tags'] = len(self._tags)
            stats['sources'] = len(self._by_source)
            
            if stats['cache_hits'] + stats['cache_misses'] > 0:
                stats['hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
            
            return stats


# ═══════════════════════════════════════════════════════════════════════════════
# TEAM MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class TeamManager:
    """
    Manages teams of agents.
    
    Handles team creation, membership, and coordination.
    """
    
    def __init__(self):
        """Initialize team manager."""
        self._teams: Dict[str, Team] = {}
        self._agent_teams: Dict[str, Set[str]] = defaultdict(set)  # agent -> teams
        
        # Statistics
        self._stats = {
            'teams_created': 0,
            'teams_dissolved': 0,
            'members_added': 0,
            'members_removed': 0,
        }
        
        self._lock = threading.RLock()
    
    def create_team(
        self,
        name: str,
        description: str = "",
        goal: str = "",
        max_members: int = 10,
    ) -> Team:
        """Create a new team"""
        with self._lock:
            team = Team(
                name=name,
                description=description,
                goal=goal,
                max_members=max_members,
            )
            
            self._teams[team.id] = team
            self._stats['teams_created'] += 1
            
            logger.info(f"Created team {team.id}: {name}")
            return team
    
    def dissolve_team(self, team_id: str) -> bool:
        """Dissolve a team"""
        with self._lock:
            team = self._teams.get(team_id)
            if not team:
                return False
            
            # Remove all members
            for agent_id in team.members:
                self._agent_teams[agent_id].discard(team_id)
            
            del self._teams[team_id]
            self._stats['teams_dissolved'] += 1
            
            logger.info(f"Dissolved team {team_id}")
            return True
    
    def add_member(
        self,
        team_id: str,
        agent_id: str,
        role: TeamRole = TeamRole.MEMBER,
    ) -> bool:
        """Add a member to a team"""
        with self._lock:
            team = self._teams.get(team_id)
            if not team:
                return False
            
            if team.add_member(agent_id, role):
                self._agent_teams[agent_id].add(team_id)
                self._stats['members_added'] += 1
                return True
            
            return False
    
    def remove_member(self, team_id: str, agent_id: str) -> bool:
        """Remove a member from a team"""
        with self._lock:
            team = self._teams.get(team_id)
            if not team:
                return False
            
            if team.remove_member(agent_id):
                self._agent_teams[agent_id].discard(team_id)
                self._stats['members_removed'] += 1
                return True
            
            return False
    
    def get_team(self, team_id: str) -> Optional[Team]:
        """Get team by ID"""
        return self._teams.get(team_id)
    
    def get_teams_for_agent(self, agent_id: str) -> List[Team]:
        """Get all teams an agent belongs to"""
        with self._lock:
            team_ids = self._agent_teams.get(agent_id, set())
            return [self._teams[tid] for tid in team_ids if tid in self._teams]
    
    def find_teams_by_capability(self, capability: str) -> List[Team]:
        """Find teams with a specific capability"""
        with self._lock:
            matching = []
            for team in self._teams.values():
                if capability in team.capabilities_required:
                    matching.append(team)
            return matching
    
    def record_task_result(self, team_id: str, success: bool):
        """Record task result for a team"""
        with self._lock:
            team = self._teams.get(team_id)
            if team:
                if success:
                    team.tasks_completed += 1
                else:
                    team.tasks_failed += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get team statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['active_teams'] = len(self._teams)
            
            total_members = sum(t.member_count for t in self._teams.values())
            stats['total_members'] = total_members
            
            if self._teams:
                stats['avg_team_size'] = total_members / len(self._teams)
            
            return stats


# ═══════════════════════════════════════════════════════════════════════════════
# EVENT BUS
# ═══════════════════════════════════════════════════════════════════════════════

class EventBus:
    """
    Event bus for publish-subscribe communication.
    
    Enables decoupled event-driven communication.
    """
    
    def __init__(self):
        """Initialize event bus."""
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._event_history: deque = deque(maxlen=1000)
        
        # Statistics
        self._stats = {
            'events_published': 0,
            'events_delivered': 0,
            'subscribers_total': 0,
        }
        
        self._lock = threading.RLock()
    
    def subscribe(
        self,
        event_type: EventType,
        callback: Callable[[Event], None],
    ) -> str:
        """
        Subscribe to events of a type.
        
        Args:
            event_type: Type of events to subscribe to
            callback: Function to call when event occurs
            
        Returns:
            Subscription ID for unsubscription
        """
        with self._lock:
            sub_id = f"sub_{uuid.uuid4().hex[:8]}"
            self._subscribers[event_type].append((sub_id, callback))
            self._stats['subscribers_total'] += 1
            
            logger.debug(f"Subscribed to {event_type.name} with {sub_id}")
            return sub_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events"""
        with self._lock:
            for event_type, subscribers in self._subscribers.items():
                for i, (sub_id, _) in enumerate(subscribers):
                    if sub_id == subscription_id:
                        subscribers.pop(i)
                        self._stats['subscribers_total'] -= 1
                        return True
            return False
    
    def publish(
        self,
        event_type: EventType,
        source: str,
        data: Dict[str, Any] = None,
        message: str = "",
        severity: str = "info",
    ) -> Event:
        """
        Publish an event.
        
        Args:
            event_type: Type of event
            source: Source of event
            data: Event data
            message: Human-readable message
            severity: Severity level
            
        Returns:
            Event object
        """
        with self._lock:
            event = Event(
                event_type=event_type,
                source=source,
                data=data or {},
                message=message,
                severity=severity,
            )
            
            # Store in history
            self._event_history.append(event)
            self._stats['events_published'] += 1
            
            # Notify subscribers
            subscribers = self._subscribers.get(event_type, [])
            for sub_id, callback in subscribers:
                try:
                    callback(event)
                    event.subscribers_notified += 1
                    self._stats['events_delivered'] += 1
                except Exception as e:
                    logger.error(f"Event callback error: {e}")
            
            event.propagated = True
            
            return event
    
    def get_history(
        self,
        event_type: EventType = None,
        source: str = None,
        limit: int = 100,
    ) -> List[Event]:
        """Get event history"""
        with self._lock:
            events = list(self._event_history)
            
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            
            if source:
                events = [e for e in events if e.source == source]
            
            return events[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['event_types_subscribed'] = len(self._subscribers)
            stats['history_size'] = len(self._event_history)
            return stats


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT COLLABORATION MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class AgentCollaborationManager:
    """
    Main manager for agent collaboration.
    
    Coordinates all collaboration components.
    """
    
    def __init__(
        self,
        node_id: str = None,
        enable_consensus: bool = True,
        enable_knowledge: bool = True,
        enable_teams: bool = True,
        enable_events: bool = True,
    ):
        """
        Initialize collaboration manager.
        
        Args:
            node_id: Unique node identifier
            enable_consensus: Enable consensus engine
            enable_knowledge: Enable shared knowledge base
            enable_teams: Enable team management
            enable_events: Enable event bus
        """
        self._node_id = node_id or f"node_{uuid.uuid4().hex[:8]}"
        
        # Components
        self._router = MessageRouter()
        self._consensus = ConsensusEngine(node_id) if enable_consensus else None
        self._knowledge = SharedKnowledgeBase() if enable_knowledge else None
        self._teams = TeamManager() if enable_teams else None
        self._events = EventBus() if enable_events else None
        
        # Agent tracking
        self._registered_agents: Dict[str, Queue] = {}
        
        # Statistics
        self._stats = {
            'agents_registered': 0,
            'messages_sent': 0,
            'collaborations': 0,
        }
        
        self._lock = threading.RLock()
        
        logger.info(f"AgentCollaborationManager initialized for {self._node_id}")
    
    def register_agent(self, agent_id: str) -> Queue:
        """Register an agent for collaboration"""
        with self._lock:
            queue = self._router.register_agent(agent_id)
            self._registered_agents[agent_id] = queue
            
            if self._consensus:
                self._consensus.set_voting_rights(agent_id, 1.0)
            
            # Publish event
            if self._events:
                self._events.publish(
                    EventType.AGENT_JOINED,
                    source=self._node_id,
                    data={'agent_id': agent_id},
                )
            
            self._stats['agents_registered'] += 1
            
            logger.info(f"Agent {agent_id} registered for collaboration")
            return queue
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        with self._lock:
            self._router.unregister_agent(agent_id)
            
            if agent_id in self._registered_agents:
                del self._registered_agents[agent_id]
            
            if self._consensus:
                self._consensus.remove_voting_rights(agent_id)
            
            if self._events:
                self._events.publish(
                    EventType.AGENT_LEFT,
                    source=self._node_id,
                    data={'agent_id': agent_id},
                )
    
    def send_message(
        self,
        sender_id: str,
        receiver_id: str,
        message_type: MessageType,
        content: Dict[str, Any],
        priority: Priority = Priority.NORMAL,
        requires_ack: bool = False,
    ) -> str:
        """Send a message between agents"""
        message = Message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            priority=priority,
            requires_ack=requires_ack,
        )
        
        recipients = self._router.route(message)
        self._stats['messages_sent'] += 1
        
        return message.id
    
    def broadcast(
        self,
        sender_id: str,
        message_type: MessageType,
        content: Dict[str, Any],
        exclude_self: bool = True,
    ) -> int:
        """Broadcast a message to all agents"""
        message = Message(
            sender_id=sender_id,
            receiver_id="*" if not exclude_self else "",
            channel="broadcast:all",
            message_type=message_type,
            content=content,
        )
        
        return self._router.route(message)
    
    def create_proposal(
        self,
        title: str,
        description: str,
        content: Dict[str, Any],
    ) -> Optional[Proposal]:
        """Create a consensus proposal"""
        if not self._consensus:
            return None
        
        return self._consensus.create_proposal(title, description, content)
    
    def vote(
        self,
        proposal_id: str,
        voter_id: str,
        vote_type: VoteType,
        reason: str = "",
    ) -> Optional[Vote]:
        """Cast a vote"""
        if not self._consensus:
            return None
        
        return self._consensus.cast_vote(proposal_id, voter_id, vote_type, reason)
    
    def store_knowledge(
        self,
        key: str,
        value: Any,
        source_agent: str,
        **kwargs,
    ) -> Optional[KnowledgeEntry]:
        """Store shared knowledge"""
        if not self._knowledge:
            return None
        
        return self._knowledge.store(key, value, source_agent, **kwargs)
    
    def get_knowledge(self, key: str) -> Optional[KnowledgeEntry]:
        """Retrieve shared knowledge"""
        if not self._knowledge:
            return None
        
        return self._knowledge.retrieve(key)
    
    def create_team(
        self,
        name: str,
        description: str = "",
        goal: str = "",
    ) -> Optional[Team]:
        """Create a new team"""
        if not self._teams:
            return None
        
        return self._teams.create_team(name, description, goal)
    
    def subscribe_to_events(
        self,
        event_type: EventType,
        callback: Callable[[Event], None],
    ) -> Optional[str]:
        """Subscribe to events"""
        if not self._events:
            return None
        
        return self._events.subscribe(event_type, callback)
    
    def publish_event(
        self,
        event_type: EventType,
        source: str,
        data: Dict[str, Any] = None,
        message: str = "",
    ) -> Optional[Event]:
        """Publish an event"""
        if not self._events:
            return None
        
        return self._events.publish(event_type, source, data, message)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['router'] = self._router.get_stats()
            
            if self._consensus:
                stats['consensus'] = self._consensus.get_stats()
            
            if self._knowledge:
                stats['knowledge'] = self._knowledge.get_stats()
            
            if self._teams:
                stats['teams'] = self._teams.get_stats()
            
            if self._events:
                stats['events'] = self._events.get_stats()
            
            return stats


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

_manager: Optional[AgentCollaborationManager] = None


def get_collaboration_manager(node_id: str = None) -> AgentCollaborationManager:
    """Get global collaboration manager"""
    global _manager
    if _manager is None:
        _manager = AgentCollaborationManager(node_id=node_id)
    return _manager


# ═══════════════════════════════════════════════════════════════════════════════
# SELF TEST
# ═══════════════════════════════════════════════════════════════════════════════

def self_test():
    """Run self-test"""
    print("\n" + "="*60)
    print("Agent Collaboration Module Test")
    print("="*60)
    
    # Create manager
    manager = AgentCollaborationManager(node_id="test_node")
    
    # Register agents
    print("\n1. Registering agents...")
    queue1 = manager.register_agent("agent_1")
    queue2 = manager.register_agent("agent_2")
    queue3 = manager.register_agent("agent_3")
    print("   Registered: agent_1, agent_2, agent_3")
    
    # Send messages
    print("\n2. Sending messages...")
    msg_id = manager.send_message(
        sender_id="agent_1",
        receiver_id="agent_2",
        message_type=MessageType.QUERY,
        content={"question": "What is your status?"},
    )
    print(f"   Message sent: {msg_id}")
    
    # Broadcast
    print("\n3. Broadcasting...")
    recipients = manager.broadcast(
        sender_id="agent_1",
        message_type=MessageType.NOTIFICATION,
        content={"message": "Hello everyone!"},
    )
    print(f"   Reached {recipients} agents")
    
    # Create proposal
    print("\n4. Creating consensus proposal...")
    proposal = manager.create_proposal(
        title="Test Proposal",
        description="Should we proceed with task X?",
        content={"task_id": "task_001"},
    )
    print(f"   Proposal ID: {proposal.id}")
    
    # Vote
    print("\n5. Voting...")
    manager.vote(proposal.id, "agent_1", VoteType.YES, "I agree")
    manager.vote(proposal.id, "agent_2", VoteType.YES, "Sounds good")
    manager.vote(proposal.id, "agent_3", VoteType.NO, "Need more info")
    print(f"   Proposal outcome: {proposal.outcome}")
    
    # Knowledge base
    print("\n6. Knowledge base...")
    entry = manager.store_knowledge(
        key="project_status",
        value={"progress": 75, "blockers": []},
        source_agent="agent_1",
        category="project",
    )
    print(f"   Stored: {entry.key}")
    
    retrieved = manager.get_knowledge("project_status")
    print(f"   Retrieved: {retrieved.value if retrieved else None}")
    
    # Teams
    print("\n7. Teams...")
    team = manager.create_team(
        name="Alpha Team",
        description="Primary development team",
    )
    manager._teams.add_member(team.id, "agent_1", TeamRole.LEADER)
    manager._teams.add_member(team.id, "agent_2", TeamRole.MEMBER)
    print(f"   Team created: {team.id} with {team.member_count} members")
    
    # Events
    print("\n8. Events...")
    def event_handler(event):
        print(f"   Event received: {event.event_type.name}")
    
    manager.subscribe_to_events(EventType.TASK_COMPLETED, event_handler)
    manager.publish_event(
        EventType.TASK_COMPLETED,
        source="agent_1",
        message="Task XYZ completed",
    )
    
    # Stats
    print("\n9. Statistics:")
    stats = manager.get_stats()
    print(f"   Agents registered: {stats['agents_registered']}")
    print(f"   Messages sent: {stats['messages_sent']}")
    print(f"   Router stats: {stats['router']['total_routed']} routed")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    self_test()
