#[derive(Clone, Copy)]
pub(crate) enum MetricId {
    DiffFramePairsTotal,
    SourceGetFrame,
    SourceFilter,
    DenoisedGetFrame,
    ConsumerRecv,
    DifferDiffFrame,
    DifferFinish,
    OutputWrite,
    ReaderSendPacket,
    ReaderReceiveFrame,
    ReaderBuildFrame,
    ReaderCopyY,
    ReaderCopyU,
    ReaderCopyV,
}

#[cfg(feature = "profile-diff")]
mod enabled {
    use std::{
        sync::atomic::{AtomicU64, Ordering},
        time::Instant,
    };

    use super::MetricId;

    struct Metric {
        nanos: AtomicU64,
        count: AtomicU64,
    }

    impl Metric {
        const fn new() -> Self {
            Self {
                nanos: AtomicU64::new(0),
                count: AtomicU64::new(0),
            }
        }
    }

    const METRIC_COUNT: usize = 14;
    static METRICS: [Metric; METRIC_COUNT] = [
        Metric::new(),
        Metric::new(),
        Metric::new(),
        Metric::new(),
        Metric::new(),
        Metric::new(),
        Metric::new(),
        Metric::new(),
        Metric::new(),
        Metric::new(),
        Metric::new(),
        Metric::new(),
        Metric::new(),
        Metric::new(),
    ];

    const LABELS: [&str; METRIC_COUNT] = [
        "diff_frame_pairs_total",
        "source_get_frame",
        "source_filter",
        "denoised_get_frame",
        "consumer_recv",
        "differ_diff_frame",
        "differ_finish",
        "output_write",
        "reader_send_packet",
        "reader_receive_frame",
        "reader_build_frame",
        "reader_copy_y",
        "reader_copy_u",
        "reader_copy_v",
    ];

    pub(crate) fn time<T>(id: MetricId, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let result = f();
        let elapsed = start.elapsed().as_nanos();
        let elapsed = elapsed.min(u128::from(u64::MAX)) as u64;
        let metric = &METRICS[id as usize];
        metric.nanos.fetch_add(elapsed, Ordering::Relaxed);
        metric.count.fetch_add(1, Ordering::Relaxed);
        result
    }

    pub(crate) fn print_report() {
        eprintln!("grav1synth diff profile:");
        eprintln!(
            "{:<30} {:>10} {:>12} {:>12}",
            "metric", "count", "total_ms", "avg_ms"
        );
        for (label, metric) in LABELS.iter().zip(METRICS.iter()) {
            let count = metric.count.load(Ordering::Relaxed);
            if count == 0 {
                continue;
            }
            let nanos = metric.nanos.load(Ordering::Relaxed);
            let total_ms = nanos as f64 / 1_000_000.0;
            let avg_ms = total_ms / count as f64;
            eprintln!("{label:<30} {count:>10} {total_ms:>12.3} {avg_ms:>12.3}");
        }
    }
}

#[cfg(feature = "profile-diff")]
pub(crate) use enabled::{print_report, time};

#[cfg(not(feature = "profile-diff"))]
#[inline(always)]
pub(crate) fn time<T>(_: MetricId, f: impl FnOnce() -> T) -> T {
    f()
}

#[cfg(not(feature = "profile-diff"))]
#[inline(always)]
pub(crate) fn print_report() {}
