# Your first week

Run `quant setup` once, then live your life. This page is the checklist the
setup command prints.

## The checklist

1. Set up the Telegram bot (quant notify-setup).
2. Install the schedule (quant schedule).
3. Make the first Monthly decision (enter your budget, approve the split).
4. Execute it in Trade Republic (savings plan and one-off buy).
5. Enter what you bought (one line), so the math stays honest.
6. Refresh your broker values if the last sync is older than 35 days.
7. Live your life.
8. Glance at the summary on Overview.
9. A Telegram alert means place the order in the broker app; it executes at market open.

## Setting up the Telegram bot (5 minutes, once)

1. Open Telegram and find **@BotFather** (the verified one).
2. Send `/newbot`, choose a name, then a username ending in `bot`.
3. BotFather replies with a **token**. Copy it. It is a secret: never share it,
   and it never enters git.
4. Open a chat with your new bot and send it any message, for example `start`.
   Without this step the next step returns nothing.
5. Get your chat id: open
   `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser and find
   `"chat":{"id":123456789}`. That number is your chat id. (Fallback: message
   @userinfobot.)
6. In a terminal: `quant notify-setup`, choose `telegram`, paste the token, paste
   the chat id. A test message arrives on your phone.

If the token ever leaks, run `/revoke` in BotFather and the token is dead.

## Entering your broker values (about 5 minutes, once a month)

Trade Republic does not export a portfolio CSV. You enter four fields per
position by hand, from the app screens. Open **My holdings**, expand
**Broker statement (editable)**, and for each position type:

1. **Symbol** (for example `5J50.DE`).
2. **Average entry price per share** (from the position's detail screen).
3. **Current value in EUR** (the position's current value).
4. **Profit in EUR** (the position's profit or loss).

A worked example, one row:

```
Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR
5J50.DE,10.00,1000.00,50.00
```

Four positions take about 5 minutes once per month. Between syncs the app shows
estimates from known shares and the latest price, labeled estimated; the broker
statement keeps the numbers you typed, labeled with the last sync date.

## Where each number comes from

- **Invested estimate.** Shares (from your last sync or a recorded buy) times the
  latest close. It moves when you record a purchase. Labeled "estimated, as of
  the date of the last sync or actuals".
- **Broker statement.** The four fields you typed, exactly as your broker
  reported them at the last sync. This is the truth for profit and loss.
- **Flows.** Buys, sells, and dividends you record. They keep the performance
  math honest: deposits do not count as profit.
- **Scores.** Structure (fundamental quality) and Tactics (timing), 0 to 100,
  recomputed on each review from market data and fundamentals.

## What to do next

1. Open the app, go to **Monthly decision**, enter your budget (for example
   200 EUR), look at the split, and approve it.
2. Execute it in Trade Republic (the savings plan and the one-off buy).
3. Return to the app and enter what you bought (one line), so the math stays
   honest.
4. If the last broker sync is older than a month, refresh your broker values
   (four fields per position, about 5 minutes).
5. Live your life. Glance at the summary on Overview. An alert arrives in
   Telegram on its own; you place the order from your phone in the evening and
   it executes at the next market open.
6. After a week, run `quant doctor`: the last daily run should be yesterday and
   the monitoring gap should be 0.
