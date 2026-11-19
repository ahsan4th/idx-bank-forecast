"""Prakiraan harga saham bank besar BEI untuk 1 sampai 5 hari bursa ke depan.

Jalankan:  streamlit run streamlit_app.py

Alur: harga harian Yahoo Finance -> validasi -> backtest rolling-origin semua model
pada origin yang sama -> model dengan MAE terendah direkomendasikan -> prakiraan
h+1..h+5 dengan interval dari residual backtest. Harga live (tertunda ~10 menit)
diperbarui oleh fragment terpisah.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

from forecaster import branding, charts, market, models, pipeline
from forecaster import data as price_data
from forecaster import sentiment as news
from forecaster.config import load_settings
from forecaster.evaluation import min_calibration_size

ROOT = Path(__file__).resolve().parent
SETTINGS = load_settings(ROOT / "settings.toml")
HORIZON = SETTINGS.forecast.horizon
TZ = SETTINGS.market.timezone

BANK_NAMES = {
    "BBCA.JK": "Bank Central Asia",
    "BBRI.JK": "Bank Rakyat Indonesia",
    "BMRI.JK": "Bank Mandiri",
    "BBNI.JK": "Bank Negara Indonesia",
}
PACKAGES = (
    "streamlit", "yfinance", "pandas", "numpy", "scikit-learn", "scipy",
    "exchange_calendars", "chronos-forecasting", "timesfm", "timestral", "torch", "gdeltdoc",
)

st.set_page_config(
    page_title="Prakiraan harga saham bank BEI",
    page_icon=":material/query_stats:",
    layout="wide",
)

# Tanda air di pojok kanan bawah, ditempelkan ke container header (key "header").
# st.html yang hanya berisi <style> tidak memakan ruang di halaman.
WATERMARK_KEY = "header"
if SETTINGS.ui.watermark:
    st.html(branding.watermark_css(SETTINGS.ui.watermark, WATERMARK_KEY))


# ------------------------------------------------------------------ formatting


def code(ticker: str) -> str:
    return ticker.removesuffix(".JK")


def rupiah(value: float) -> str:
    return "Rp" + f"{value:,.0f}".replace(",", ".")


def signed_pct(value: float) -> str:
    return f"{value:+.2f}%".replace(".", ",")


def clock(ts: pd.Timestamp) -> str:
    return ts.tz_convert(TZ).strftime("%H:%M")


DAY_NAMES = ("Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min")


def day_label(ts: pd.Timestamp) -> str:
    return f"{DAY_NAMES[ts.dayofweek]} {ts.day} {ts.strftime('%b')}"


# ------------------------------------------------------------------ hard requirements

missing = [m for m in ("yfinance", "exchange_calendars", "sklearn") if not models.module_available(m)]
if missing:
    st.error(
        f"Paket wajib belum terpasang: {', '.join(missing)}. "
        "Jalankan `pip install -r requirements.txt`, lalu muat ulang halaman.",
        icon=":material/error:",
    )
    st.stop()


def _expected_fetch_errors() -> tuple[type[BaseException], ...]:
    """Errors that data sources raise for bad responses, rate limits, or the network.

    These are shown per ticker. Anything else (a bug) is left to crash with a traceback.
    """
    errors: list[type[BaseException]] = [ValueError, ImportError, OSError, TimeoutError]
    from yfinance.exceptions import YFException

    errors.append(YFException)
    if models.module_available("requests"):
        import requests

        errors.append(requests.exceptions.RequestException)
    if models.module_available("curl_cffi"):
        from curl_cffi.requests.exceptions import RequestException as CurlRequestException

        errors.append(CurlRequestException)
    return tuple(errors)


FETCH_ERRORS = _expected_fetch_errors()


# ------------------------------------------------------------------ cached data and models


@st.cache_resource(show_spinner=False)
def exchange_calendar(name: str) -> market.TradingCalendar:
    return market.load_exchange_calendar(name)


@st.cache_resource(max_entries=4, show_spinner=False)
def foundation_runner(key: str, fs) -> models.FoundationRunner:  # noqa: ANN001 - frozen dataclass
    return models.load_foundation_runner(key, fs)


def load_runner(key: str) -> models.FoundationRunner:
    return foundation_runner(key, SETTINGS.models.foundation)


@st.cache_data(ttl=SETTINGS.data.daily_cache_ttl_seconds, max_entries=40, show_spinner=False)
def daily_bars(ticker: str, period: str) -> pd.DataFrame:
    return price_data.fetch_daily_history(ticker, period, SETTINGS.data, SETTINGS.market)


@st.cache_data(ttl=SETTINGS.data.intraday_cache_ttl_seconds, max_entries=20, show_spinner=False)
def intraday_bars(ticker: str) -> pd.DataFrame:
    return price_data.fetch_intraday(ticker, SETTINGS.data, SETTINGS.market)


@st.cache_data(max_entries=40, show_spinner=False)
def backtest(
    ticker: str,
    final: pd.DataFrame,
    model_keys: tuple[str, ...],
    n_origins: int,
    sentiment_final: pd.DataFrame | None,
) -> pipeline.BacktestResult:
    return pipeline.run_backtest(
        ticker, final, model_keys, SETTINGS, n_origins, load_runner, sentiment_final, None
    )


def _gdelt_fetch(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    from gdeltdoc import Filters, GdeltDoc

    s = SETTINGS.sentiment
    return news.fetch_gdelt_daily(
        s.query_for(ticker), start, end, s.chunk_days, s.request_pause_seconds,
        GdeltDoc(), Filters, None,
    )


@st.cache_data(max_entries=100, show_spinner=False)
def gdelt_past_year(ticker: str, year: int) -> pd.DataFrame:
    """A finished calendar year of GDELT tone (cached without expiry)."""
    return _gdelt_fetch(ticker, pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31))


@st.cache_data(ttl=SETTINGS.sentiment.cache_ttl_seconds, max_entries=20, show_spinner=False)
def gdelt_current_year(ticker: str, start: str, end: str) -> pd.DataFrame:
    return _gdelt_fetch(ticker, pd.Timestamp(start), pd.Timestamp(end))


@st.cache_data(ttl=900, max_entries=20, show_spinner=False)
def gdelt_headlines(ticker: str) -> pd.DataFrame:
    from gdeltdoc import Filters, GdeltDoc

    s = SETTINGS.sentiment
    return news.recent_articles(
        s.query_for(ticker), GdeltDoc(), Filters, s.headlines_timespan, s.headlines_max_records
    )


def gdelt_daily(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    parts = [gdelt_past_year(ticker, y) for y in range(start.year, end.year)]
    parts.append(gdelt_current_year(ticker, f"{end.year}-01-01", end.strftime("%Y-%m-%d")))
    daily = pd.concat(parts)
    return daily.loc[(daily.index >= start) & (daily.index <= end)]


# ------------------------------------------------------------------ sidebar

state = st.session_state
state.setdefault("forecast_inputs", {})
# Top-level code runs only on full reruns, so this counts them. The live fragment
# compares forecast inputs only after the current full run has recorded them;
# otherwise a changed input would trigger st.rerun() again before being recorded.
state["full_runs"] = state.get("full_runs", 0) + 1
availability = models.model_availability()
gdelt_ready = models.module_available("gdeltdoc")
all_tickers = list(SETTINGS.data.tickers_default + SETTINGS.data.tickers_optional)
model_options = [k for k in models.MODEL_SPECS if k != models.NAIVE_KEY]


def model_option_label(key: str) -> str:
    label = models.MODEL_SPECS[key].label
    return label if availability[key] else f"{label} (belum terpasang)"


with st.sidebar:
    with st.form("run_settings", border=False):
        st.markdown("**Data dan model**")
        tickers = st.multiselect(
            "Saham",
            all_tickers,
            default=list(SETTINGS.data.tickers_default),
            format_func=lambda t: f"{code(t)}, {BANK_NAMES[t]}",
        )
        period = st.selectbox(
            "Riwayat harga harian",
            SETTINGS.data.history_period_options,
            index=SETTINGS.data.history_period_options.index(SETTINGS.data.history_period),
        )
        chosen = st.multiselect(
            "Model",
            model_options,
            default=[k for k in SETTINGS.models.default_selection if k in model_options and availability[k]],
            format_func=model_option_label,
            help="Random walk selalu ikut sebagai tolok ukur.",
        )
        n_origins = st.select_slider(
            "Origin backtest",
            options=SETTINGS.backtest.n_origins_options,
            value=SETTINGS.backtest.n_origins_default,
            help="Jumlah hari bursa terakhir yang dipakai menguji semua model dengan aturan yang sama.",
        )
        use_sentiment = st.toggle(
            "Sentimen berita sebagai input",
            value=SETTINGS.sentiment.enabled_default and gdelt_ready,
            disabled=not gdelt_ready,
            help="Tone harian GDELT sebagai fitur gradient boosting dan past covariate "
            "Chronos-2/Timestral. Setiap model dibandingkan dengan dan tanpa sentimen. "
            + ("" if gdelt_ready else "Butuh `pip install gdeltdoc`."),
        )
        st.form_submit_button("Terapkan", type="primary", icon=":material/play_arrow:", width="stretch")

    st.markdown("**Tampilan**")
    # A split-conformal interval at level L needs a minimum number of residuals;
    # only levels the chosen backtest can support are offered.
    valid_levels = [
        lv for lv in SETTINGS.forecast.interval_levels if min_calibration_size(lv) <= n_origins
    ]
    level = st.segmented_control(
        "Interval prakiraan",
        valid_levels,
        default=(
            SETTINGS.forecast.interval_level_default
            if SETTINGS.forecast.interval_level_default in valid_levels
            else max(valid_levels)
        ),
        required=True,
        format_func=lambda x: f"{x:.0%}",
    )
    hidden_levels = [lv for lv in SETTINGS.forecast.interval_levels if lv not in valid_levels]
    if hidden_levels:
        st.caption(
            "; ".join(
                f"Interval {lv:.0%} butuh minimal {min_calibration_size(lv)} origin backtest"
                for lv in hidden_levels
            )
            + "."
        )
    live_origin = st.toggle(
        "Pakai harga live sebagai close sementara",
        value=False,
        help="Mati: prakiraan dihitung dari close final terakhir, jadi saat bursa buka h+1 adalah "
        "close hari ini. Hidup: harga live diperlakukan sebagai close hari ini dan h+1 adalah "
        "hari bursa berikutnya. Interval tetap dihitung dari residual close final.",
    )
    refresh = st.select_slider(
        "Perbarui harga live",
        options=SETTINGS.ui.refresh_options_seconds,
        value=SETTINGS.ui.refresh_default_seconds,
        format_func=lambda s: "mati" if s == 0 else f"{s} detik",
    )

unavailable = [k for k in chosen if not availability[k]]
active = [k for k in chosen if availability[k]]
model_keys = [models.NAIVE_KEY, *active]
if use_sentiment:
    model_keys += [models.variant_key(k, True) for k in active if models.MODEL_SPECS[k].supports_sentiment]
model_keys_t = tuple(model_keys)

# ------------------------------------------------------------------ header

now = pd.Timestamp.now(tz=TZ)
calendar = exchange_calendar(SETTINGS.market.calendar)
today = pd.Timestamp(now.date())
phase = market.market_phase(now, calendar.is_session(today), SETTINGS.market)
trading_now = phase in market.TRADING_PHASES
mode = st.context.theme.type or "light"

with st.container(horizontal=True, vertical_alignment="center", key=WATERMARK_KEY):
    st.title("Prakiraan harga saham bank BEI", icon=":material/query_stats:", anchor=False)
    st.badge(
        market.PHASE_LABELS_ID[phase],
        icon=":material/radio_button_checked:" if trading_now else ":material/schedule:",
        color="green" if trading_now else "gray",
    )
st.caption(
    f"Harga dari Yahoo Finance, tertunda sekitar {SETTINGS.market.yahoo_delay_minutes} menit "
    f"untuk BEI. Prakiraan 1 sampai {HORIZON} hari bursa ke depan, diuji dengan backtest "
    "rolling-origin terhadap random walk. Bukan rekomendasi investasi."
)

if not tickers:
    st.info("Pilih minimal satu saham di sidebar, lalu tekan **Terapkan**.", icon=":material/info:")
    st.stop()
for key in unavailable:
    spec = models.MODEL_SPECS[key]
    st.warning(
        f"{spec.label} belum terpasang dan dilewati. Pasang dengan `{spec.install_hint}`.",
        icon=":material/extension_off:",
    )

# ------------------------------------------------------------------ live quotes (fragment)


def forecast_inputs_now(ticker: str) -> tuple[str, float | None]:
    """What the forecast of `ticker` depends on right now: last final date, live price."""
    clean, _ = price_data.validate_daily(daily_bars(ticker, period), ticker, SETTINGS.data)
    final, _ = price_data.split_provisional(clean, pd.Timestamp.now(tz=TZ), SETTINGS.market)
    live = None
    if live_origin:
        bar = price_data.provisional_bar_from_intraday(intraday_bars(ticker), pd.Timestamp(pd.Timestamp.now(tz=TZ).date()))
        if bar is not None and bar.index[0] > final.index[-1]:
            live = float(bar["Close"].iloc[0])
    return str(final.index[-1].date()), live


@st.fragment(run_every=refresh if refresh else None)
def live_panel(ticker_list: list[str]) -> None:
    with st.container(horizontal=True):
        for ticker in ticker_list:
            try:
                quote = price_data.build_live_quote(ticker, intraday_bars(ticker), daily_bars(ticker, period))
            except FETCH_ERRORS as exc:
                st.warning(f"{code(ticker)}: harga live tidak tersedia ({type(exc).__name__}: {exc})")
                continue
            if quote.change == 0:
                delta, description = 0, "tidak berubah"
            else:
                delta, description = signed_pct(quote.change_pct), f"vs close {day_label(quote.previous_close_date)}"
            st.metric(
                f"{code(ticker)}, {BANK_NAMES[ticker]}",
                rupiah(quote.price),
                delta,
                delta_description=description,
                chart_data=quote.intraday_close.round(0).tolist(),
                chart_type="area",
                border=True,
                help=f"Bar terakhir {clock(quote.bar_time)} WIB, {day_label(quote.session_date)}.",
            )
    st.caption(
        f"Diperbarui {pd.Timestamp.now(tz=TZ):%H:%M:%S} WIB"
        + (f", otomatis tiap {refresh} detik." if refresh else ". Pembaruan otomatis mati.")
    )
    # A new final close (or, in live mode, a new live price) changes the forecasts:
    # rerun the whole app so they are recomputed from the new data.
    if state.get("inputs_recorded_in") != state["full_runs"]:
        return
    for ticker in ticker_list:
        recorded = state["forecast_inputs"].get(ticker)
        if recorded is None:
            continue
        try:
            current = forecast_inputs_now(ticker)
        except FETCH_ERRORS:
            continue  # the quote card above already shows the fetch problem
        if current != recorded:
            st.rerun()


live_panel(tickers)

# ------------------------------------------------------------------ forecasts per ticker

results: dict[str, dict] = {}
errors: dict[str, str] = {}
with st.status("Menyiapkan data dan backtest", expanded=False) as status:
    for ticker in tickers:
        status.update(label=f"Menyiapkan {code(ticker)}")
        try:
            clean, quality = price_data.validate_daily(daily_bars(ticker, period), ticker, SETTINGS.data)
            final, provisional_daily = price_data.split_provisional(clean, now, SETTINGS.market)
            origin_frame, live_price = final, None
            if live_origin:
                bar = price_data.provisional_bar_from_intraday(intraday_bars(ticker), today)
                if bar is not None and bar.index[0] > final.index[-1]:
                    origin_frame = pd.concat([final, bar])
                    live_price = float(bar["Close"].iloc[0])

            sent_final = sent_origin = None
            if use_sentiment:
                s = SETTINGS.sentiment
                coverage_start = max(pd.Timestamp(s.earliest_date), final.index[0] - pd.Timedelta(days=10))
                last_complete_utc_day = pd.Timestamp(now.tz_convert("UTC").date()) - pd.Timedelta(days=1)
                st.write(f"{code(ticker)}: mengambil tone berita GDELT (sekali per tahun data, bisa beberapa menit)")
                daily_tone = gdelt_daily(ticker, coverage_start, last_complete_utc_day)
                sent_origin = news.align_to_trading_days(
                    daily_tone, origin_frame.index, s.lag_days, s.ewm_halflife_days, coverage_start
                )
                sent_final = sent_origin.loc[final.index]

            st.write(f"{code(ticker)}: backtest {len(model_keys_t)} model pada {n_origins} origin")
            bt = backtest(ticker, final, model_keys_t, n_origins, sent_final)
            forecast = pipeline.run_final_forecast(
                ticker, final, origin_frame, bt, SETTINGS, level, calendar, load_runner,
                sent_final, sent_origin,
            )
        except FETCH_ERRORS as exc:
            errors[ticker] = f"{type(exc).__name__}: {exc}"
            continue
        results[ticker] = {
            "final": final,
            "origin_frame": origin_frame,
            "provisional_daily": provisional_daily,
            "live_price": live_price,
            "quality": quality,
            "bt": bt,
            "forecast": forecast,
            "sentiment": sent_origin,
        }
        state["forecast_inputs"][ticker] = (str(final.index[-1].date()), live_price)
    state["inputs_recorded_in"] = state["full_runs"]
    status.update(
        label="Data dan backtest siap" if results else "Tidak ada prakiraan yang bisa dihitung",
        state="complete" if results else "error",
    )

for ticker, message in errors.items():
    st.error(f"{code(ticker)}: {message}", icon=":material/error:")
if not results:
    st.stop()

# ------------------------------------------------------------------ summary

date_sets = {tuple(sorted(res["forecast"].table["date"].unique())) for res in results.values()}
st.subheader("Prakiraan model terbaik per saham", anchor=False)
if len(date_sets) == 1:
    dates_text = ", ".join(
        f"h+{i + 1} = {day_label(pd.Timestamp(d))}" for i, d in enumerate(next(iter(date_sets)))
    )
else:
    dates_text = "Tanggal h+1..h+5 berbeda antarsaham karena tanggal origin berbeda; lihat detail"
st.caption(f"Model terbaik = MAE rata-rata h+1..h+{HORIZON} terendah di backtest. {dates_text}.")
summary_rows = []
for ticker, r in results.items():
    bt, fc = r["bt"], r["forecast"]
    best = bt.best_model
    best_row = bt.summary.set_index("model").loc[best]
    row = {
        "Saham": code(ticker),
        "Harga origin": fc.origin_price,
        "Tanggal origin": fc.origin_date,
        "Model terbaik": models.variant_label(best),
        "Skill vs random walk (%)": best_row["skill_pct"],
    }
    best_table = fc.table[fc.table["model"] == best].set_index("h")
    for h in range(1, HORIZON + 1):
        row[f"h+{h}"] = best_table.loc[h, "median"]
    summary_rows.append(row)
st.dataframe(
    pd.DataFrame(summary_rows),
    hide_index=True,
    column_config={
        "Harga origin": st.column_config.NumberColumn(format="localized"),
        "Tanggal origin": st.column_config.DateColumn(format="D MMM YYYY"),
        "Skill vs random walk (%)": st.column_config.NumberColumn(
            format="%+.1f", help="1 - MAE model / MAE random walk, rata-rata h+1..h+5."
        ),
        **{f"h+{h}": st.column_config.NumberColumn(format="localized") for h in range(1, HORIZON + 1)},
    },
    alt="Ringkasan prakiraan lima hari bursa ke depan per saham dengan model terbaik",
)

# ------------------------------------------------------------------ detail

st.subheader("Detail per saham", anchor=False)
view = st.segmented_control(
    "Saham yang ditampilkan",
    list(results),
    default=next(iter(results)),
    required=True,
    format_func=code,
    label_visibility="collapsed",
)
r = results[view]
bt, fc = r["bt"], r["forecast"]
ranked = list(bt.summary["model"])

for key, message in {**bt.failures, **fc.failures}.items():
    st.error(f"{models.variant_label(key)} gagal dan tidak diikutkan: {message}", icon=":material/error:")
if fc.provisional:
    st.info(
        f"Origin memakai harga live {rupiah(fc.origin_price)} sebagai close sementara hari ini. "
        "h+1 adalah hari bursa berikutnya.",
        icon=":material/bolt:",
    )
if not fc.origin_is_session:
    st.warning(
        f"Tanggal data {fc.origin_date.date()} bukan hari bursa menurut kalender {SETTINGS.market.calendar}; "
        "periksa data atau kalender libur.",
        icon=":material/event_busy:",
    )

with st.container(horizontal=True, vertical_alignment="bottom"):
    focus = st.selectbox(
        "Model di grafik",
        ranked,
        index=0,
        format_func=models.variant_label,
        help="Urut dari MAE backtest terendah.",
    )
    compare_all = st.toggle("Tampilkan median model lain", value=False)

chart_col, table_col = st.columns([3, 2], gap="medium")
focus_rows = fc.table[fc.table["model"] == focus]
with chart_col.container(border=True, height="stretch"):
    compare = (
        {k: (models.variant_label(k), g) for k, g in fc.table.groupby("model") if k != focus}
        if compare_all
        else {}
    )
    history = r["origin_frame"]["Close"].iloc[-SETTINGS.ui.chart_history_days:]
    st.echarts_chart(
        charts.fan_chart_option(
            history, focus, models.variant_label(focus), focus_rows, compare, level, mode,
            SETTINGS.ui.watermark,
        ),
        height=380,
        alt=f"Harga {code(view)} {len(history)} hari bursa terakhir dan prakiraan "
        f"{models.variant_label(focus)} {HORIZON} hari ke depan dengan interval {level:.0%}",
    )

with table_col.container(border=True, height="stretch"):
    st.markdown(f"**{models.variant_label(focus)}**, origin {rupiah(fc.origin_price)} ({day_label(fc.origin_date)})")
    shown = focus_rows.assign(
        Horizon=[f"h+{h}" for h in focus_rows["h"]],
    )[["Horizon", "date", "median", "lower", "upper", "change_pct"]]
    st.dataframe(
        shown,
        hide_index=True,
        column_config={
            "date": st.column_config.DateColumn("Tanggal", format="ddd, D MMM"),
            "median": st.column_config.NumberColumn("Prakiraan", format="localized"),
            "lower": st.column_config.NumberColumn(f"Bawah {level:.0%}", format="localized"),
            "upper": st.column_config.NumberColumn(f"Atas {level:.0%}", format="localized"),
            "change_pct": st.column_config.NumberColumn("Perubahan (%)", format="%+.2f"),
        },
        alt=f"Tabel prakiraan {code(view)} h+1 sampai h+{HORIZON}",
    )
    row = bt.summary.set_index("model").loc[focus]
    if focus == models.NAIVE_KEY:
        st.caption("Random walk adalah tolok ukurnya sendiri: prakiraan = harga origin.")
    elif row["n_sig_better"] > 0:
        st.caption(
            f"Lebih akurat dari random walk secara signifikan (DM, koreksi Holm, alpha "
            f"{SETTINGS.backtest.dm_alpha}) pada {int(row['n_sig_better'])} dari {HORIZON} horizon."
        )
    else:
        st.caption(
            f"Belum ada horizon yang signifikan lebih akurat dari random walk (DM, koreksi Holm, "
            f"alpha {SETTINGS.backtest.dm_alpha}). Perlakukan angka ini sebagai skenario, bukan sinyal."
        )
    if trading_now and not fc.provisional and r["live_price"] is None:
        h1 = focus_rows[focus_rows["h"] == 1].iloc[0]
        if pd.Timestamp(h1["date"]) == today:
            try:
                live = price_data.build_live_quote(view, intraday_bars(view), r["final"]).price
            except FETCH_ERRORS:
                live = None  # optional comparison; the live panel shows the error
            if live is not None:
                where = "di dalam" if h1["lower"] <= live <= h1["upper"] else ("di atas" if live > h1["upper"] else "di bawah")
                st.caption(f"Harga live {rupiah(live)} saat ini {where} interval h+1 (close hari ini).")

# ------------------------------------------------------------------ backtest evaluation

st.subheader("Evaluasi backtest", anchor=False)
st.caption(
    f"{len(bt.origin_dates)} origin, {bt.origin_dates[0]} sampai {bt.origin_dates[-1]}; semua model "
    "dinilai pada origin yang sama. MAE relatif di bawah 1 berarti lebih akurat dari random walk."
)
metrics = bt.metrics
rel = metrics.pivot(index="model", columns="h", values="rel_mae").loc[ranked]
labels = {k: models.variant_label(k) for k in ranked}
eval_chart, eval_table = st.columns([2, 3], gap="medium")
with eval_chart.container(border=True, height="stretch"):
    st.echarts_chart(
        charts.skill_chart_option(rel, labels, mode, SETTINGS.ui.watermark),
        height=320,
        alt=f"MAE relatif terhadap random walk per horizon untuk {code(view)}; model terbaik "
        f"{models.variant_label(bt.best_model)}",
    )
with eval_table.container(border=True, height="stretch"):
    mae = metrics.pivot(index="model", columns="h", values="mae").loc[ranked]
    table = pd.DataFrame(
        {
            "Model": [labels[k] for k in ranked],
            **{f"MAE h+{h}": mae[h].to_numpy() for h in range(1, HORIZON + 1)},
            "MAE relatif": bt.summary.set_index("model").loc[ranked, "mean_rel_mae"].to_numpy(),
            "Signifikan lebih baik": bt.summary.set_index("model").loc[ranked, "n_sig_better"].to_numpy(),
            "Arah benar h+1 (%)": metrics[metrics["h"] == 1].set_index("model").loc[ranked, "hit_rate"].to_numpy() * 100,
            f"Cakupan interval {level:.0%} (OOS)": bt.summary.set_index("model").loc[
                ranked, f"conformal_cov_{int(round(level * 100))}"
            ].to_numpy() * 100,
        }
    )
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            **{f"MAE h+{h}": st.column_config.NumberColumn(format="localized") for h in range(1, HORIZON + 1)},
            "MAE relatif": st.column_config.NumberColumn(format="%.3f"),
            "Signifikan lebih baik": st.column_config.NumberColumn(
                help=f"Jumlah horizon dengan uji Diebold-Mariano (HLN, koreksi Holm) p < {SETTINGS.backtest.dm_alpha} dan model lebih akurat."
            ),
            "Arah benar h+1 (%)": st.column_config.NumberColumn(format="%.1f", help="Hari dengan harga tidak berubah tidak dihitung."),
            f"Cakupan interval {level:.0%} (OOS)": st.column_config.NumberColumn(
                format="%.1f",
                help="Interval dikalibrasi pada paruh awal origin, cakupan diukur pada paruh akhir. Kosong jika origin terlalu sedikit.",
            ),
        },
        alt=f"Tabel evaluasi backtest {code(view)} per model",
    )

with st.expander("Rincian metrik per horizon"):
    detail = metrics.assign(Model=metrics["model"].map(models.variant_label))
    st.dataframe(
        detail[["Model", "h", "mae", "rmse", "mape", "rel_mae", "dm_stat", "dm_p", "dm_p_holm",
                "dm_note", "hits", "trials", "hit_rate", "hit_lo", "hit_hi", "native_cov"]],
        hide_index=True,
        column_config={
            "mae": st.column_config.NumberColumn("MAE", format="localized"),
            "rmse": st.column_config.NumberColumn("RMSE", format="localized"),
            "mape": st.column_config.NumberColumn("MAPE (%)", format="%.2f"),
            "rel_mae": st.column_config.NumberColumn("MAE relatif", format="%.3f"),
            "dm_stat": st.column_config.NumberColumn("Statistik DM", format="%.2f"),
            "dm_p": st.column_config.NumberColumn("p DM", format="%.4f"),
            "dm_p_holm": st.column_config.NumberColumn("p DM (Holm)", format="%.4f"),
            "dm_note": "Catatan DM",
            "hit_rate": st.column_config.NumberColumn("Arah benar", format="percent"),
            "hit_lo": st.column_config.NumberColumn("CI bawah", format="percent"),
            "hit_hi": st.column_config.NumberColumn("CI atas", format="percent"),
            "native_cov": st.column_config.NumberColumn(
                "Cakupan kuantil model 10-90%", format="percent",
                help="Hanya model foundation: porsi harga aktual di dalam kuantil 10% s/d 90% milik model.",
            ),
        },
        alt=f"Metrik backtest lengkap {code(view)} per model dan horizon",
    )

if len(bt.ablation):
    with st.container(border=True):
        st.markdown("**Ablation sentimen berita**")
        st.caption(
            "Rasio < 1 berarti sentimen menurunkan galat. p dari uji DM antara model dengan dan tanpa "
            "sentimen, dikoreksi Holm."
        )
        ab = bt.ablation.assign(Model=bt.ablation["model"].map(models.variant_label))
        st.dataframe(
            ab[["Model", "h", "mae_without", "mae_with", "ratio", "dm_p", "dm_p_holm"]],
            hide_index=True,
            column_config={
                "mae_without": st.column_config.NumberColumn("MAE tanpa", format="localized"),
                "mae_with": st.column_config.NumberColumn("MAE dengan", format="localized"),
                "ratio": st.column_config.NumberColumn("Rasio", format="%.3f"),
                "dm_p": st.column_config.NumberColumn("p DM", format="%.4f"),
                "dm_p_holm": st.column_config.NumberColumn("p DM (Holm)", format="%.4f"),
            },
            alt=f"Perbandingan galat dengan dan tanpa sentimen untuk {code(view)}",
        )

if r["sentiment"] is not None:
    with st.container(border=True):
        st.markdown("**Tone berita GDELT**")
        st.caption(
            "Tone per hari bursa dari bin UTC sampai H-1 (tanpa kebocoran waktu). Kosong berarti tidak ada "
            "artikel atau di luar cakupan data."
        )
        st.echarts_chart(
            charts.tone_chart_option(
                r["sentiment"].iloc[-SETTINGS.ui.chart_history_days:], mode, SETTINGS.ui.watermark
            ),
            height=240,
            alt=f"Tone berita harian untuk {code(view)} dan rata-rata tertimbang eksponensialnya",
        )
        try:
            headlines = gdelt_headlines(view)
        except FETCH_ERRORS as exc:
            st.warning(f"Judul berita terbaru tidak tersedia ({type(exc).__name__}: {exc})")
        else:
            if headlines.empty:
                st.caption("Tidak ada berita yang cocok dalam beberapa hari terakhir.")
            else:
                st.dataframe(
                    headlines,
                    hide_index=True,
                    column_config={
                        "seen": st.column_config.DatetimeColumn("Terlihat (UTC)", format="D MMM, HH:mm"),
                        "title": st.column_config.TextColumn("Judul", width="large"),
                        "domain": "Sumber",
                        "language": "Bahasa",
                        "sourcecountry": "Negara",
                        "url": st.column_config.LinkColumn("Tautan", display_text="buka"),
                    },
                    alt=f"Judul berita terbaru tentang {code(view)} dari GDELT",
                )

with st.expander("Kualitas data harga"):
    q = r["quality"]
    st.caption(
        f"{q.n_clean} bar harian ({q.first_date} sampai {q.last_date}) dari {q.n_raw} bar mentah. "
        + ("Bar hari ini belum final dan tidak dipakai sebagai close." if r["provisional_daily"] is not None else "")
    )
    issues = q.issues()
    if issues:
        for note in issues:
            st.markdown(f"- {note}")
    else:
        st.caption("Tidak ada temuan.")

# ------------------------------------------------------------------ method and reproducibility


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "tidak terpasang"
    return versions


with st.expander("Metodologi dan reproduksibilitas"):
    st.markdown(
        f"""
**Data.** Harga harian Yahoo Finance lewat `yfinance`, disesuaikan dividen dan stock split.
Bar hari ini baru dianggap final setelah {SETTINGS.market.daily_bar_final_time:%H:%M} WIB.

**Model.** Random walk (tolok ukur), gradient boosting (HistGradientBoosting, satu model per
horizon untuk return log kumulatif, refit tiap {SETTINGS.models.gbm.refit_every} origin), dan model
foundation zero-shot (Chronos-2, Chronos-Bolt, Timestral, TimesFM 2.5) dengan konteks
{SETTINGS.models.foundation.context_length} hari bursa. Ensembel = median model non-sentimen.

**Backtest.** Origin = {n_origins} hari bursa terakhir yang kelima horizonnya sudah teramati. Tidak
ada model yang melihat data setelah origin-nya. Uji Diebold-Mariano (koreksi Harvey-Leybourne-Newbold,
loss galat absolut) terhadap random walk, koreksi Holm atas semua model x horizon.

**Interval.** Split-conformal dari residual log backtest per horizon. Jaminan cakupannya mengasumsikan
residual exchangeable, yang hanya berlaku kira-kira pada saham (volatilitas mengelompok); cakupan
out-of-sample ditampilkan di tabel evaluasi.

**Keterbatasan.** Pemilihan model terbaik memakai backtest yang sama dengan pelaporannya, jadi skill
model terbaik sedikit optimistis. Model foundation mungkin pernah melihat sebagian riwayat saham ini
saat pra-pelatihan, sehingga backtest periode lama bisa optimistis. Harga saham harian mendekati
random walk; kegagalan mengalahkan random walk adalah hasil yang sah.
"""
    )
    run_meta = {
        "generated_at": now.isoformat(),
        "tickers": tickers,
        "history_period": period,
        "models": list(model_keys_t),
        "n_origins": n_origins,
        "interval_level": level,
        "live_origin": live_origin,
        "sentiment": use_sentiment,
        "packages": package_versions(),
        "settings": asdict(SETTINGS),
    }
    forecasts_all = pd.concat(
        [res["forecast"].table.assign(ticker=t, origin_date=res["forecast"].origin_date) for t, res in results.items()]
    )
    metrics_all = pd.concat([res["bt"].metrics.assign(ticker=t) for t, res in results.items()])
    with st.container(horizontal=True):
        st.download_button(
            "Prakiraan (CSV)", forecasts_all.to_csv(index=False), "prakiraan.csv", "text/csv",
            icon=":material/download:",
        )
        st.download_button(
            "Metrik backtest (CSV)", metrics_all.to_csv(index=False), "backtest_metrics.csv", "text/csv",
            icon=":material/download:",
        )
        st.download_button(
            "Metadata run (JSON)", json.dumps(run_meta, indent=2, default=str), "run_metadata.json",
            "application/json", icon=":material/download:",
        )
    st.json(run_meta["packages"], expanded=False)
