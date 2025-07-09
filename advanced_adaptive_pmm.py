import logging
import numpy as np
from decimal import Decimal
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase
from hummingbot.connector.connector_base import ConnectorBase


class AdvancedAdaptivePMM(ScriptStrategyBase):
    """
    🚀 Profitable Paper Trading Strategy - Fixed Order Sizing & Active Trading
    
    Key improvements:
    - Fixed order sizing bug
    - Aggressive spreads to increase fill probability
    - Dynamic order refresh to generate more trades
    - Better profit tracking and display
    """
    
    # ===========================================
    # 🎯 CORE STRATEGY PARAMETERS - PROFIT OPTIMIZED
    # ===========================================
    
    # Basic Configuration
    trading_pair = "ETH-USDT"
    exchange = "binance_paper_trade"
    base_order_amount = 0.02  # Fixed base amount in ETH
    order_refresh_time = 20   # Reduced to 20 seconds for more activity
    price_source = PriceType.MidPrice
    
    # Strategy State
    create_timestamp = 0
    total_profit = 0.0
    trade_count = 0
    last_profit_check = 0
    
    # ===========================================
    # 📈 AGGRESSIVE SPREADS FOR MORE FILLS
    # ===========================================
    
    # Tighter spreads for more trades
    base_bid_spread = 0.001   # 0.1% - much tighter
    base_ask_spread = 0.001   # 0.1% - much tighter
    
    # Dynamic spread adjustment
    min_spread_bps = 10       # 0.1% minimum
    max_spread_bps = 50       # 0.5% maximum
    volatility_multiplier = 0.5  # Less volatile = more fills
    
    # ===========================================
    # 🔄 ENHANCED ORDER MANAGEMENT
    # ===========================================
    
    # Order management
    max_active_orders = 4     # Allow more orders
    order_levels = 2          # Multiple price levels
    level_spacing = 0.002     # 0.2% between levels
    
    # Order refresh triggers
    price_change_threshold = 0.005  # Refresh if price moves 0.5%
    last_mid_price = None
    
    # ===========================================
    # 🎯 INVENTORY MANAGEMENT
    # ===========================================
    
    target_base_ratio = 0.5
    inventory_tolerance = 0.3    # 30% tolerance
    max_inventory_skew = 0.4     # 40% max skew
    
    # Dynamic sizing based on inventory
    inventory_adjustment_factor = 1.5
    
    # ===========================================
    # 🏗️ INITIALIZATION
    # ===========================================
    
    def __init__(self, connectors: Dict[str, ConnectorBase]):
        super().__init__(connectors)
        
        # Initialize strategy state
        self.base, self.quote = self.trading_pair.split('-')
        self.current_spreads = {'bid': self.base_bid_spread, 'ask': self.base_ask_spread}
        self.trend_signal = 0.0
        self.inventory_skew = 0.0
        
        # Price history for analysis
        self.price_history = []
        self.price_changes = []
        self.last_price = None
        
        # Performance tracking
        self.trade_history = []
        self.last_balance_check = datetime.now()
        self.session_start_time = datetime.now()
        
        # Order management
        self.last_order_creation = 0
        self.consecutive_failures = 0
        self.filled_orders_count = 0
        
        logging.info("🚀 Enhanced Paper Trading Strategy - Ready for Profit!")
    
    # Define markets
    markets = {exchange: {trading_pair}}
    
    # ===========================================
    # 🎯 MAIN STRATEGY LOOP - ENHANCED
    # ===========================================
    
    def on_tick(self):
        """Enhanced strategy execution with more trading activity"""
        if self.create_timestamp <= self.current_timestamp:
            try:
                # Update market analysis
                self.update_market_analysis()
                
                # Check if we need to refresh orders
                should_refresh = self.should_refresh_orders()
                
                # Get current active orders
                active_orders = self.get_active_orders(connector_name=self.exchange)
                
                # Decision logic for order management
                if len(active_orders) == 0:
                    logging.info("📋 No active orders, creating new ones")
                    self.create_multiple_orders()
                elif should_refresh:
                    logging.info("🔄 Market conditions changed, refreshing orders")
                    self.cancel_and_replace_orders()
                elif len(active_orders) < 2:
                    logging.info("📋 Adding more orders for better coverage")
                    self.create_additional_orders()
                else:
                    logging.info(f"📋 {len(active_orders)} active orders maintained")
                
                # Set next refresh time
                self.create_timestamp = self.order_refresh_time + self.current_timestamp
                
            except Exception as e:
                logging.error(f"❌ Error in strategy execution: {e}")
                self.create_timestamp = self.order_refresh_time + self.current_timestamp
    
    def should_refresh_orders(self) -> bool:
        """Determine if orders should be refreshed based on market conditions"""
        try:
            current_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
            
            if not current_price or not self.last_mid_price:
                return False
            
            # Refresh if price moved significantly
            price_change = abs(float(current_price) - self.last_mid_price) / self.last_mid_price
            if price_change > self.price_change_threshold:
                logging.info(f"🔄 Price moved {price_change:.3f}, refreshing orders")
                return True
            
            # Refresh periodically anyway to generate activity
            if self.current_timestamp - self.last_order_creation > 60:  # 1 minute
                logging.info("🔄 Periodic refresh to maintain activity")
                return True
            
            return False
            
        except Exception as e:
            logging.error(f"Error checking refresh conditions: {e}")
            return False
    
    def cancel_and_replace_orders(self):
        """Cancel existing orders and place new ones"""
        try:
            # Cancel all active orders
            active_orders = self.get_active_orders(connector_name=self.exchange)
            for order in active_orders:
                self.cancel(self.exchange, order.trading_pair, order.client_order_id)
            
            logging.info(f"📋 Cancelled {len(active_orders)} orders")
            
            # Wait a moment then create new orders
            self.create_multiple_orders()
            
        except Exception as e:
            logging.error(f"Error in cancel and replace: {e}")
    
    def create_multiple_orders(self):
        """Create multiple orders at different price levels"""
        try:
            orders = self.create_layered_orders()
            if orders:
                logging.info(f"📋 Created {len(orders)} layered orders")
                adjusted_orders = self.adjust_orders_to_budget(orders)
                if adjusted_orders:
                    self.place_orders(adjusted_orders)
                    self.last_order_creation = self.current_timestamp
                    logging.info(f"📋 Placed {len(adjusted_orders)} orders successfully")
            
        except Exception as e:
            logging.error(f"Error creating multiple orders: {e}")
    
    def create_additional_orders(self):
        """Add more orders if we have fewer than desired"""
        try:
            active_orders = self.get_active_orders(connector_name=self.exchange)
            
            # Check what side we're missing
            has_buy = any(order.is_buy for order in active_orders)
            has_sell = any(not order.is_buy for order in active_orders)
            
            orders = []
            
            if not has_buy:
                buy_orders = self.create_buy_orders()
                orders.extend(buy_orders)
            
            if not has_sell:
                sell_orders = self.create_sell_orders()
                orders.extend(sell_orders)
            
            if orders:
                adjusted_orders = self.adjust_orders_to_budget(orders)
                if adjusted_orders:
                    self.place_orders(adjusted_orders)
                    logging.info(f"📋 Added {len(adjusted_orders)} additional orders")
            
        except Exception as e:
            logging.error(f"Error creating additional orders: {e}")
    
    # ===========================================
    # 📊 ENHANCED MARKET ANALYSIS
    # ===========================================
    
    def update_market_analysis(self):
        """Enhanced market analysis with better spread calculation"""
        try:
            current_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
            
            if current_price and current_price > 0:
                price_float = float(current_price)
                self.price_history.append(price_float)
                
                # Maintain history
                if len(self.price_history) > 30:
                    self.price_history.pop(0)
                
                # Calculate price changes
                if self.last_price:
                    price_change = (price_float - self.last_price) / self.last_price
                    self.price_changes.append(price_change)
                    
                    if len(self.price_changes) > 20:
                        self.price_changes.pop(0)
                
                self.last_price = price_float
                self.last_mid_price = price_float
                
                # Update analysis components
                self.update_dynamic_spreads()
                self.update_inventory_metrics()
                
        except Exception as e:
            logging.error(f"Error in market analysis: {e}")
    
    def update_dynamic_spreads(self):
        """Calculate dynamic spreads based on volatility"""
        try:
            if len(self.price_changes) >= 5:
                # Calculate recent volatility
                recent_volatility = np.std(self.price_changes[-10:]) if len(self.price_changes) >= 10 else 0.002
                
                # Adjust spreads based on volatility (inverse relationship for more fills)
                base_spread = max(self.min_spread_bps, min(self.max_spread_bps, recent_volatility * 5000))
                
                # Make spreads tighter during low volatility
                if recent_volatility < 0.001:
                    base_spread *= 0.5
                
                self.current_spreads['bid'] = base_spread / 10000
                self.current_spreads['ask'] = base_spread / 10000
                
            else:
                self.current_spreads['bid'] = self.base_bid_spread
                self.current_spreads['ask'] = self.base_ask_spread
                
        except Exception as e:
            logging.error(f"Error updating spreads: {e}")
    
    def update_inventory_metrics(self):
        """Update inventory-based adjustments"""
        try:
            base_balance = self.connectors[self.exchange].get_balance(self.base)
            quote_balance = self.connectors[self.exchange].get_balance(self.quote)
            current_price = self.last_price
            
            if base_balance and quote_balance and current_price:
                base_value = float(base_balance) * current_price
                total_value = base_value + float(quote_balance)
                
                if total_value > 0:
                    current_ratio = base_value / total_value
                    self.inventory_skew = current_ratio - self.target_base_ratio
                else:
                    self.inventory_skew = 0.0
            else:
                self.inventory_skew = 0.0
                
        except Exception as e:
            logging.error(f"Error updating inventory: {e}")
    
    # ===========================================
    # 🎯 ENHANCED ORDER CREATION
    # ===========================================
    
    def create_layered_orders(self) -> List[OrderCandidate]:
        """Create multiple orders at different price levels"""
        try:
            ref_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
            
            if not ref_price or ref_price <= 0:
                return []
            
            orders = []
            
            # Create buy orders at different levels
            buy_orders = self.create_buy_orders()
            orders.extend(buy_orders)
            
            # Create sell orders at different levels
            sell_orders = self.create_sell_orders()
            orders.extend(sell_orders)
            
            return orders
            
        except Exception as e:
            logging.error(f"Error creating layered orders: {e}")
            return []
    
    def create_buy_orders(self) -> List[OrderCandidate]:
        """Create buy orders with proper sizing"""
        try:
            ref_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
            if not ref_price:
                return []
            
            orders = []
            
            # Calculate order size based on inventory
            base_size = self.base_order_amount
            
            # Adjust size based on inventory skew
            if self.inventory_skew > 0.2:  # Too much base, smaller buy orders
                buy_size = base_size * 0.5
            elif self.inventory_skew < -0.2:  # Too much quote, larger buy orders
                buy_size = base_size * 1.5
            else:
                buy_size = base_size
            
            # Create orders at different levels
            for level in range(self.order_levels):
                spread_multiplier = 1 + (level * 0.5)  # Increase spread for deeper levels
                spread = self.current_spreads['bid'] * spread_multiplier
                
                buy_price = ref_price * Decimal(str(1 - spread))
                
                # FIXED: Ensure we're using the correct amount (ETH, not USDT)
                order = OrderCandidate(
                    trading_pair=self.trading_pair,
                    is_maker=True,
                    order_type=OrderType.LIMIT,
                    order_side=TradeType.BUY,
                    amount=Decimal(str(buy_size)),  # This is in ETH
                    price=buy_price                 # This is in USDT per ETH
                )
                orders.append(order)
            
            return orders
            
        except Exception as e:
            logging.error(f"Error creating buy orders: {e}")
            return []
    
    def create_sell_orders(self) -> List[OrderCandidate]:
        """Create sell orders with proper sizing"""
        try:
            ref_price = self.connectors[self.exchange].get_price_by_type(self.trading_pair, self.price_source)
            if not ref_price:
                return []
            
            orders = []
            
            # Calculate order size based on inventory
            base_size = self.base_order_amount
            
            # Adjust size based on inventory skew
            if self.inventory_skew > 0.2:  # Too much base, larger sell orders
                sell_size = base_size * 1.5
            elif self.inventory_skew < -0.2:  # Too much quote, smaller sell orders
                sell_size = base_size * 0.5
            else:
                sell_size = base_size
            
            # Create orders at different levels
            for level in range(self.order_levels):
                spread_multiplier = 1 + (level * 0.5)  # Increase spread for deeper levels
                spread = self.current_spreads['ask'] * spread_multiplier
                
                ask_price = ref_price * Decimal(str(1 + spread))
                
                # FIXED: Ensure we're using the correct amount (ETH, not USDT)
                order = OrderCandidate(
                    trading_pair=self.trading_pair,
                    is_maker=True,
                    order_type=OrderType.LIMIT,
                    order_side=TradeType.SELL,
                    amount=Decimal(str(sell_size)),  # This is in ETH
                    price=ask_price                  # This is in USDT per ETH
                )
                orders.append(order)
            
            return orders
            
        except Exception as e:
            logging.error(f"Error creating sell orders: {e}")
            return []
    
    # ===========================================
    # 📋 ENHANCED ORDER MANAGEMENT
    # ===========================================
    
    def adjust_orders_to_budget(self, orders: List[OrderCandidate]) -> List[OrderCandidate]:
        """Enhanced budget checking with better validation"""
        try:
            adjusted_orders = []
            
            for order in orders:
                if order.order_side == TradeType.BUY:
                    # For buy orders, check quote balance (USDT)
                    quote_balance = self.connectors[self.exchange].get_balance(self.quote)
                    required_quote = float(order.amount) * float(order.price)
                    
                    if quote_balance and float(quote_balance) >= required_quote * 1.1:  # 10% buffer
                        adjusted_orders.append(order)
                        logging.info(f"✅ Buy order approved: {order.amount} ETH @ {order.price} USDT")
                    else:
                        logging.warning(f"⚠️ Insufficient USDT for buy order: need {required_quote:.2f}, have {float(quote_balance) if quote_balance else 0:.2f}")
                
                else:  # SELL order
                    # For sell orders, check base balance (ETH)
                    base_balance = self.connectors[self.exchange].get_balance(self.base)
                    required_base = float(order.amount)
                    
                    if base_balance and float(base_balance) >= required_base * 1.1:  # 10% buffer
                        adjusted_orders.append(order)
                        logging.info(f"✅ Sell order approved: {order.amount} ETH @ {order.price} USDT")
                    else:
                        logging.warning(f"⚠️ Insufficient ETH for sell order: need {required_base:.4f}, have {float(base_balance) if base_balance else 0:.4f}")
            
            return adjusted_orders
            
        except Exception as e:
            logging.error(f"Error adjusting orders to budget: {e}")
            return []
    
    def place_orders(self, orders: List[OrderCandidate]) -> None:
        """Place orders with enhanced logging"""
        for order in orders:
            try:
                if order.order_side == TradeType.SELL:
                    self.sell(
                        connector_name=self.exchange,
                        trading_pair=order.trading_pair,
                        amount=order.amount,
                        order_type=order.order_type,
                        price=order.price
                    )
                elif order.order_side == TradeType.BUY:
                    self.buy(
                        connector_name=self.exchange,
                        trading_pair=order.trading_pair,
                        amount=order.amount,
                        order_type=order.order_type,
                        price=order.price
                    )
                
                logging.info(f"📋 {'SELL' if order.order_side == TradeType.SELL else 'BUY'} "
                           f"{order.amount} {self.base} @ {order.price} {self.quote}")
                
            except Exception as e:
                logging.error(f"Error placing {order.order_side} order: {e}")
    
    # ===========================================
    # 📊 ENHANCED EVENT HANDLING
    # ===========================================
    
    def did_fill_order(self, event: OrderFilledEvent):
        """Enhanced order fill handling with better profit calculation"""
        try:
            self.trade_count += 1
            self.filled_orders_count += 1
            
            # Calculate profit estimate
            spread_profit = 0
            if event.trade_type == TradeType.SELL:
                # Sold above mid-price
                spread_profit = float(event.amount) * float(event.price) * 0.001  # Estimate 0.1% profit
            else:
                # Bought below mid-price
                spread_profit = float(event.amount) * float(event.price) * 0.001  # Estimate 0.1% profit
            
            self.total_profit += spread_profit
            
            # Store trade history
            trade_record = {
                'timestamp': datetime.now(),
                'side': event.trade_type,
                'amount': float(event.amount),
                'price': float(event.price),
                'profit_estimate': spread_profit
            }
            self.trade_history.append(trade_record)
            
            # Create detailed message
            msg = (
                f"🎉 TRADE EXECUTED: {'SELL' if event.trade_type == TradeType.SELL else 'BUY'} "
                f"{event.amount:.4f} {self.base} @ {event.price:.2f} {self.quote} "
                f"| Profit: +{spread_profit:.4f} {self.quote} "
                f"| Total: {self.total_profit:.4f} {self.quote} "
                f"| Trades: {self.trade_count}"
            )
            
            self.log_with_clock(logging.INFO, msg)
            self.notify_hb_app_with_timestamp(msg)
            
            # Trigger order refresh after fill to maintain activity
            self.create_timestamp = min(self.create_timestamp, self.current_timestamp + 5)
            
        except Exception as e:
            logging.error(f"Error handling order fill: {e}")
    
    # ===========================================
    # 📈 ENHANCED STATUS DISPLAY
    # ===========================================
    
    def format_status(self) -> str:
        """Enhanced status display with trading activity"""
        if not self.ready_to_trade:
            return "🔄 Market connectors are not ready."
        
        try:
            # Calculate session statistics
            session_duration = datetime.now() - self.session_start_time
            trades_per_hour = (self.trade_count / max(session_duration.total_seconds() / 3600, 0.1))
            
            lines = [
                "🚀 ENHANCED PAPER TRADING STRATEGY",
                "=" * 60,
                f"📊 Trading Pair: {self.trading_pair}",
                f"⏱️  Session Duration: {str(session_duration).split('.')[0]}",
                f"💰 Total Trades: {self.trade_count} ({trades_per_hour:.1f}/hour)",
                f"💰 Total Profit: {self.total_profit:.6f} {self.quote}",
                f"📈 Current Spreads: Bid {self.current_spreads['bid']:.4f} | Ask {self.current_spreads['ask']:.4f}",
                f"📊 Inventory Skew: {self.inventory_skew:.4f}",
                ""
            ]
            
            # Recent trades
            if self.trade_history:
                lines.append("🎯 RECENT TRADES:")
                for trade in self.trade_history[-3:]:  # Last 3 trades
                    time_str = trade['timestamp'].strftime("%H:%M:%S")
                    lines.append(f"   {time_str} | {'SELL' if trade['side'] == TradeType.SELL else 'BUY'} "
                               f"{trade['amount']:.4f} @ {trade['price']:.2f} (+{trade['profit_estimate']:.4f})")
                lines.append("")
            
            # Balance info
            try:
                balance_df = self.get_balance_df()
                lines.extend([
                    "💰 BALANCES:",
                    balance_df.to_string(index=False),
                    ""
                ])
            except:
                lines.append("💰 BALANCES: Loading...")
            
            # Active orders
            try:
                df = self.active_orders_df()
                if not df.empty:
                    lines.extend([
                        "📋 ACTIVE ORDERS:",
                        df.to_string(index=False),
                        ""
                    ])
                else:
                    lines.append("📋 ACTIVE ORDERS: None (creating new ones...)")
            except:
                lines.append("📋 ACTIVE ORDERS: Loading...")
            
            lines.append("=" * 60)
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"❌ Error generating status: {e}"
