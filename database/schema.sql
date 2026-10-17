
CREATE TABLE articles (id SERIAL PRIMARY KEY, title TEXT, body TEXT, source TEXT, url TEXT, date DATE, sector TEXT, label TEXT, created_at TIMESTAMP DEFAULT NOW());
CREATE TABLE predictions (id SERIAL PRIMARY KEY, article_id INT REFERENCES articles(id), model TEXT, label TEXT, confidence REAL, signals TEXT, created_at TIMESTAMP DEFAULT NOW());
CREATE TABLE nepse_daily (id SERIAL PRIMARY KEY, date DATE UNIQUE NOT NULL, open DECIMAL(10,2), high DECIMAL(10,2), low DECIMAL(10,2), close DECIMAL(10,2), volume BIGINT, created_at TIMESTAMP DEFAULT NOW());
CREATE TABLE backtest_results (id SERIAL PRIMARY KEY, strategy TEXT, start_date DATE, end_date DATE, final_equity DECIMAL(15,2), sharpe REAL, max_drawdown REAL, win_rate REAL, created_at TIMESTAMP DEFAULT NOW());
CREATE INDEX idx_predictions_article ON predictions(article_id);
CREATE INDEX idx_nepse_daily_date ON nepse_daily(date);
