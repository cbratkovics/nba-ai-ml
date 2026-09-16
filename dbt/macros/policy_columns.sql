{#- Column lists of the two bronze copies of reports/policy_<season>.json, typed, so both
    render an identical empty relation before the artifact exists (see product_columns.sql). -#}
{% macro policy_report_columns() %}
    {{ return([
        ['source_file', 'varchar'], ['season', 'varchar'], ['policy_commit', 'varchar'],
        ['model_revision', 'varchar'], ['model_commit', 'varchar'], ['in_sample', 'boolean'],
        ['min_coverage', 'double'], ['population', 'varchar'], ['target', 'varchar'],
        ['n', 'integer'], ['threshold', 'double'], ['n_called', 'integer'], ['coverage', 'double'],
        ['n_resolved', 'integer'], ['n_push', 'integer'], ['n_hit', 'integer'], ['hit_rate', 'double'],
        ['coin_flip_hit_rate', 'double'], ['coin_flip_half_width_95', 'double'],
        ['season_mean_same_rows_n', 'integer'], ['season_mean_same_rows_hit_rate', 'double'],
        ['model_beats_both', 'boolean'], ['verdict', 'varchar'],
        ['band_q10', 'double'], ['band_q25', 'double'], ['band_q75', 'double'], ['band_q90', 'double'],
        ['band_coverage_50', 'double'], ['band_coverage_80', 'double'],
    ]) }}
{% endmacro %}

{% macro policy_curve_columns() %}
    {{ return([
        ['source_file', 'varchar'], ['season', 'varchar'], ['model_revision', 'varchar'],
        ['population', 'varchar'], ['target', 'varchar'], ['threshold', 'double'],
        ['n_called', 'integer'], ['coverage', 'double'], ['n_resolved', 'integer'], ['n_push', 'integer'],
        ['n_hit', 'integer'], ['hit_rate', 'double'], ['net_correct', 'integer'],
        ['season_mean_same_rows_n', 'integer'], ['season_mean_same_rows_hit_rate', 'double'],
        ['season_mean_own_n_called', 'integer'], ['season_mean_own_hit_rate', 'double'],
    ]) }}
{% endmacro %}
