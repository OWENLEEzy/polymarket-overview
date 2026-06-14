create table if not exists search_runs (
  id text primary key,
  topic text not null,
  provider text not null,
  search_terms_json text not null,
  created_at text not null
);

create table if not exists raw_markets (
  id text primary key,
  search_run_id text not null references search_runs(id) on delete cascade,
  market_id text not null,
  payload_json text not null,
  created_at text not null
);

create table if not exists structured_markets (
  id text primary key,
  search_run_id text not null references search_runs(id) on delete cascade,
  market_id text not null,
  payload_json text not null,
  created_at text not null
);

create table if not exists structured_market_reviews (
  id text primary key,
  search_run_id text not null references search_runs(id) on delete cascade,
  payload_json text not null,
  created_at text not null
);

create table if not exists aggregation_runs (
  id text primary key,
  search_run_id text not null references search_runs(id) on delete cascade,
  object_name text not null,
  input_json text not null,
  result_json text not null,
  created_at text not null
);

create index if not exists idx_raw_markets_search_run_id on raw_markets(search_run_id);
create index if not exists idx_structured_markets_search_run_id on structured_markets(search_run_id);
create index if not exists idx_structured_market_reviews_search_run_id on structured_market_reviews(search_run_id);
create index if not exists idx_aggregation_runs_search_run_id on aggregation_runs(search_run_id);
