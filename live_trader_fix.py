                    self._interruptible_sleep(min(wait, 300))  # Check every 5 min max

            except Exception as e:
                self.logger.error(f"Unexpected error in main loop: {e}")
                if self.running:
                    self._interruptible_sleep(60)

        self._shutdown()

    def _interruptible_sleep(self, seconds: float):
        """Sleep that can be interrupted by setting self.running = False."""
        end_time = time.time() + seconds
        while self.running and time.time() < end_time:
            time.sleep(min(1, end_time - time.time()))

    def _shutdown(self):
        """Graceful shutdown: log final state."""
        self.logger.info("=" * 60)
        self.logger.info("SHUTTING DOWN PAPER TRADER")
        self.logger.info(f"  Total cycles: {self.cycle_count}")
        self.logger.info(f"  Total PnL: ${self.total_pnl:+,.2f}")
        self.logger.info(f"  Open positions: {len(self.risk_manager.positions)}")

        if self.risk_manager.positions:
            self.logger.info("  Open positions at shutdown:")
            for sym, pos in self.risk_manager.positions.items():
                self.logger.info(
                    f"    {sym}: {pos.quantity} shares @ ${pos.entry_price:.2f}"
                )

        self.self_improver._save_state()
        self.logger.info("State saved. Goodbye.")
        self.logger.info("=" * 60)

    def get_status(self) -> Dict:
        """Return current trader status."""
        risk = self.risk_manager.get_status()
        account = self.get_account_info()
        return {
            "cycle_count": self.cycle_count,
            "total_pnl": self.total_pnl,
            "current_regime": self.current_regime.value,
            "market_open": self.is_market_open(),
            "alpaca_connected": self.alpaca_connected,
            "dry_run": self.dry_run,
            "risk": risk,
            "strategy_weights": dict(self.self_improver.weights),
            "account": account,
        }


# ── Standalone entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Live Paper Trader")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate trades without hitting Alpaca API")
    args = parser.parse_args()

    trader = LiveTrader(dry_run=args.dry_run)
    trader.run()
