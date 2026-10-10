-- Belt and braces for ADR-0030: no row flagged as a bet may have its prediction after its
-- line or its line after tip-off, whatever fct_paper_bet's eligibility logic says.
select game_date, source, book, market, player_id
from {{ ref('fct_paper_bet') }}
where is_bet and not (prediction_before_line and line_before_tipoff)
