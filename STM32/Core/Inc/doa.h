/**
  ******************************************************************************
  * @file    doa.h
  * @brief   SRP-PHAT azimuth direction-of-arrival estimation for an
  *          eight-element uniform circular microphone array on STM32H753.
  *
  * Input   Interleaved 16-bit ADC samples in acquisition rank order
  *         [ch0, ch1, ... ch7, ch0, ...], supplied one half-buffer per call.
  * Output  Azimuth in degrees over [0, 360), with a peak-to-mean confidence
  *         ratio and a validity flag.
  *
  * Method  Per-frame real FFT of all channels, PHAT-weighted cross-spectra
  *         for every microphone pair, band-limited inverse transform to
  *         obtain generalised cross-correlations, and a steered-response
  *         summation over a uniform azimuth grid. The sequential conversion
  *         order of the shared ADC is corrected as a constant per-pair lag
  *         offset in the steering table.
  *
  * Bearings are referenced to the microphone at ring position 0.
  ******************************************************************************
  */

#ifndef DOA_H
#define DOA_H

#include <stdint.h>
#include "arm_math.h"

/* ===================== Array and acquisition parameters ================== */
/* DOA_N and DOA_FS must match the acquisition configuration; a mismatch is
 * not detectable at build time and invalidates every predicted delay.       */

#define DOA_N            1024        /* samples per channel per frame        */
#define DOA_NUM_MIC      8           /* elements on the ring                 */
#define DOA_FS           25000.0f    /* per-channel sample rate [Hz]         */
#define DOA_RADIUS       0.095f      /* array radius [m]                     */
#define DOA_SOUND_SPEED  343.0f      /* speed of sound [m/s]                 */

/* ===================== Processing band =================================== */
/* The upper edge is bounded by spatial aliasing. Adjacent elements are
 * separated by the chord 2*R*sin(pi/M) = 72.7 mm, giving an ambiguity limit
 * of c/(2d) = 2.36 kHz; above this no pair provides an unambiguous phase
 * difference. Antipodal pairs alias from 903 Hz, but their ambiguities fall
 * at different bearings from those of the shorter baselines and are
 * suppressed by the steered-response summation.
 *
 * The lower edge excludes supply-related and thermal low-frequency content
 * together with any residual offset left by per-frame mean subtraction.     */

#define DOA_F_LO         200.0f      /* lower band edge [Hz]                 */
#define DOA_F_HI         2000.0f     /* upper band edge [Hz]                 */

/* ===================== Estimator configuration =========================== */

#define DOA_NUM_ANGLES   360         /* azimuth grid points (1 degree steps) */

/* Peak-to-mean ratio of the accumulated response map above which a bearing
 * is reported as valid. Determined empirically.                             */
#define DOA_CONF_GATE    3.0f

/* Inter-channel acquisition skew, expressed as the interval between
 * consecutive ranks of the shared converter: (sampling + conversion cycles)
 * divided by the ADC kernel clock. Currently 32.5 + 8.5 cycles at 50 MHz.
 * Must be updated if either the sampling time or the ADC clock changes.     */
#define DOA_SKEW_DT      (41.0f / 50000000.0f)   /* [s] per rank             */

/* ===================== Result type ======================================= */

typedef struct {
    float   azimuth_deg;   /* estimated bearing over [0, 360)                */
    float   confidence;    /* peak-to-mean ratio of the response map         */
    uint8_t valid;         /* non-zero if confidence exceeds DOA_CONF_GATE   */
} doa_result_t;

/* ===================== Interface ========================================= */

/**
 * @brief  Initialise the estimator.
 *
 * Precomputes the analysis window, the microphone pair list, the band bin
 * limits and the steering delay table, and initialises the transform
 * instance. Must be called once after clocks and peripherals are configured
 * and before the first call to DOA_Process.
 */
void DOA_Init(void);

/**
 * @brief  Process one acquisition frame and produce a bearing estimate.
 *
 * @param  interleaved  One half of the acquisition buffer, containing
 *                      DOA_N * DOA_NUM_MIC samples in rank order.
 * @param  out          Receives the azimuth, confidence and validity flag.
 *
 * Executes in the caller's context. Must not be invoked from the DMA
 * interrupt handler: the handler should set a flag and the main loop should
 * call this function in response.
 */
void DOA_Process(const uint16_t *interleaved, doa_result_t *out);

/**
 * @brief  Discard the accumulated response map.
 *
 * Forces re-acquisition rather than waiting for the exponential average to
 * migrate. Intended for use when the source is known to have moved.
 */
void DOA_Reset(void);

/**
 * @brief  Verify the transform library on the target.
 *
 * Transforms a sinusoid of known frequency and records the resulting
 * magnitude peak index in g_peak_idx, which must equal 8. Confirms that the
 * floating-point unit is active, that the transform tables are linked, and
 * that the library headers are consistent with the compiled objects.
 */
void doa_smoke_test(void);

/* ===================== Instrumentation =================================== */
/* Populated for external inspection; not part of the operational interface. */

/* Steering table assertions, evaluated once by DOA_Init. Predicted values
 * follow from the array geometry: dbg_max_lag from (2R/c)*FS, and
 * dbg_geo_antisym from the antisymmetry of the propagation model, which must
 * vanish to within floating-point rounding.                                 */
extern volatile float dbg_max_lag;
extern volatile float dbg_pair0_at0;
extern volatile float dbg_diam_at0;
extern volatile float dbg_antisym_err;
extern volatile float dbg_geo_antisym;

/* Per-frame estimate, prior to temporal averaging. */
extern volatile float dbg_frame_az;
extern volatile float dbg_frame_conf;

/* Circular statistics over the most recent block of per-frame estimates.
 * dbg_cluster_R is the resultant length, which approaches 1/sqrt(N) for
 * uniformly distributed estimates and unity for perfectly concentrated ones.
 * dbg_hist holds the corresponding distribution in 10 degree bins.          */
extern volatile float    dbg_cluster_R;
extern volatile float    dbg_spread_deg;
extern volatile float    dbg_mean_az;
extern volatile uint16_t dbg_hist[36];

/* Transform self-test result. */
extern volatile uint32_t  g_peak_idx;
extern volatile float32_t g_peak_val;

#endif /* DOA_H */
