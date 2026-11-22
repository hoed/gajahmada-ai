-- GajahMada 2.0 — Sovereign Hybrid Knowledge & Ledger Schema for Supabase (PostgreSQL + pgvector)
-- Run once in Supabase SQL Editor or via psql.

create extension if not exists vector;

create table if not exists public.knowledge_records (
  content_hash text primary key,
  content text not null,
  embedding vector(384),
  source_type text default 'cognitive_brain_qa_update',
  source_url text default '',
  origin_node_id text default 'local',
  signature text default '',
  confidence double precision default 0.75,
  verification_status text default 'UNVERIFIED',
  visibility text default 'COLLECTIVE',
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create index if not exists idx_knowledge_records_visibility
  on public.knowledge_records (visibility);

create index if not exists idx_knowledge_records_created_at
  on public.knowledge_records (created_at desc);

create table if not exists public.ledger_blocks (
  block_index integer primary key,
  timestamp double precision not null,
  task_hash text not null,
  node_id text not null,
  previous_hash text not null,
  signature text default '',
  event_type text default 'TASK_COMPLETED',
  event_ref text default '',
  nonce integer default 0,
  block_hash text not null,
  created_at timestamptz default now()
);

-- Cosine similarity semantic search RPC (384-d embeddings)
create or replace function public.match_knowledge(
  query_embedding vector(384),
  match_count int default 5,
  min_similarity float default 0.15
)
returns table (
  content_hash text,
  content text,
  source_type text,
  source_url text,
  origin_node_id text,
  confidence double precision,
  verification_status text,
  visibility text,
  similarity float
)
language sql stable
as $$
  select
    kr.content_hash,
    kr.content,
    kr.source_type,
    kr.source_url,
    kr.origin_node_id,
    kr.confidence,
    kr.verification_status,
    kr.visibility,
    1 - (kr.embedding <=> query_embedding) as similarity
  from public.knowledge_records kr
  where kr.embedding is not null
    and 1 - (kr.embedding <=> query_embedding) >= min_similarity
  order by kr.embedding <=> query_embedding
  limit match_count;
$$;

alter table public.knowledge_records enable row level security;
alter table public.ledger_blocks enable row level security;

do $$
begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'public' and tablename = 'knowledge_records' and policyname = 'Allow public read of collective knowledge'
  ) then
    create policy "Allow public read of collective knowledge"
      on public.knowledge_records for select
      using (visibility = 'COLLECTIVE');
  end if;

  if not exists (
    select 1 from pg_policies
    where schemaname = 'public' and tablename = 'ledger_blocks' and policyname = 'Allow public read of ledger blocks'
  ) then
    create policy "Allow public read of ledger blocks"
      on public.ledger_blocks for select
      using (true);
  end if;
end $$;
