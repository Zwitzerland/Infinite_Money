#!/usr/bin/env python3
"""
AlphaQuanta CLI Runner - Quantum-hybrid trading agent execution.
Supports paper/live trading with quantum-enhanced alpha discovery.
"""

import asyncio
import click
import yaml
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from alphaquanta import LeanCoreAgent
from alphaquanta.telemetry.acu_tracker import ACUTracker
from alphaquanta.telemetry.qpu_tracker import QPUTracker


@click.command()
@click.option('--mode', type=click.Choice(['paper', 'live', 'backtest']), default='paper',
              help='Trading mode')
@click.option('--quantum', type=click.Choice(['on', 'off']), default='on',
              help='Enable/disable quantum modules')
@click.option('--acu-cap', type=int, default=20,
              help='Maximum ACU budget for run')
@click.option('--symbol', default='SPY',
              help='Trading symbol')
@click.option('--start', type=str, default='2018-01-01',
              help='Backtest start date (YYYY-MM-DD)')
@click.option('--end', type=str, default='2024-12-31',
              help='Backtest end date (YYYY-MM-DD)')
@click.option('--config', type=click.Path(exists=True), default='qconfig.yaml',
              help='Quantum configuration file')
@click.option('--verbose', '-v', is_flag=True,
              help='Verbose logging')
def main(mode: str, quantum: str, acu_cap: int, symbol: str, start: str, end: str,
         config: str, verbose: bool):
    """AlphaQuanta quantum-hybrid trading agent."""
    
    click.echo(f"🚀 AlphaQuanta Starting - Mode: {mode}, Quantum: {quantum}")
    
    config_path = Path(config)
    if not config_path.exists():
        click.echo(f"❌ Config file not found: {config}")
        return 1
        
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    quantum_enabled = quantum == 'on' and cfg.get('quantum', {}).get('enabled', False)
    
    if quantum_enabled:
        ibm_token = os.getenv('IBM_QUANTUM_TOKEN')
        dwave_token = os.getenv('DWAVE_API_TOKEN')
        
        if not ibm_token and not dwave_token:
            click.echo("⚠️  No quantum tokens found. Falling back to classical mode.")
            quantum_enabled = False
        elif verbose:
            click.echo("✅ Quantum tokens detected")
    
    acu_tracker = ACUTracker(budget=acu_cap)
    qpu_tracker = QPUTracker(
        budget_minutes=cfg.get('quantum', {}).get('governance', {}).get('max_qpu_minutes_per_run', 10)
    ) if quantum_enabled else None
    
    try:
        result = asyncio.run(run_trading_session(
            mode=mode,
            quantum_enabled=quantum_enabled,
            symbol=symbol,
            start_date=start,
            end_date=end,
            config=cfg,
            acu_tracker=acu_tracker,
            qpu_tracker=qpu_tracker,
            verbose=verbose
        ))
        
        click.echo(f"\n📊 Session Results:")
        click.echo(f"   ACU Used: {acu_tracker.total_used}/{acu_cap}")
        if qpu_tracker:
            click.echo(f"   QPU Time: {qpu_tracker.total_used:.2f}/{qpu_tracker.budget} minutes")
        click.echo(f"   Status: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}")
        
        if result.get('sharpe_ratio'):
            click.echo(f"   Sharpe Ratio: {result['sharpe_ratio']:.3f}")
        if result.get('quantum_uplift'):
            click.echo(f"   Quantum Uplift: +{result['quantum_uplift']:.3f}")
            
        return 0 if result['success'] else 1
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        return 1


async def run_trading_session(mode: str, quantum_enabled: bool, symbol: str,
                            start_date: str, end_date: str, config: Dict[str, Any],
                            acu_tracker: ACUTracker, qpu_tracker: Optional[Any],
                            verbose: bool) -> Dict[str, Any]:
    """Execute trading session with quantum-enhanced alpha discovery."""
    
    start_time = time.time()
    
    agent = LeanCoreAgent(
        mode=mode,
        quantum_enabled=quantum_enabled,
        config=config,
        acu_tracker=acu_tracker,
        qpu_tracker=qpu_tracker
    )
    
    if verbose:
        click.echo(f"🤖 Agent initialized - Quantum: {quantum_enabled}")
    
    if mode == 'backtest':
        result = await agent.run_backtest(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )
    elif mode == 'paper':
        result = await agent.run_paper_trading(symbol=symbol)
    else:  # live
        result = await agent.run_live_trading(symbol=symbol)
    
    elapsed_time = time.time() - start_time
    
    return {
        'success': result.get('success', False),
        'sharpe_ratio': result.get('sharpe_ratio'),
        'quantum_uplift': result.get('quantum_uplift'),
        'elapsed_time': elapsed_time,
        'acu_used': acu_tracker.total_used,
        'qpu_used': qpu_tracker.total_used if qpu_tracker else 0
    }


if __name__ == '__main__':
    main()
