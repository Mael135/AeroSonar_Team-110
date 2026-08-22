/**
  ******************************************************************************
  * @file    doa.c
  * @brief   SRP-PHAT azimuth estimation for an eight-element uniform circular
  *          microphone array on STM32H7 with CMSIS-DSP.
  *
  * Per-frame processing chain:
  *   de-interleave -> per-channel DC removal -> Hann window
  *   -> real FFT per channel
  *   -> per pair: band-limited PHAT cross-spectrum -> inverse FFT -> lag crop
  *   -> steered-response summation over the azimuth grid
  *   -> exponential averaging of the response map
  *   -> peak search with parabolic refinement
  *
  * Frame geometry, array parameters and band limits are configured in doa.h.
  * Bearings are referenced to the microphone at ring position 0; the mapping
  * from ring position to acquisition channel is defined by MIC_CH below.
  ******************************************************************************
  */

#include "doa.h"
#include <math.h>
#include <string.h>
#include "arm_math.h"

/* ===================== Derived constants ================================= */

#define NMIC      DOA_NUM_MIC
#define NFFT      DOA_N
#define NHALF     (DOA_N / 2)
#define NPAIRS    (NMIC * (NMIC - 1) / 2)

/* Maximum inter-microphone delay, equal to the array diameter divided by the
 * speed of sound. Bounds the correlation lag window; three samples of guard
 * are added for fractional interpolation near the extremes.                 */
#define MAX_LAG_S   (2.0f * DOA_RADIUS / DOA_SOUND_SPEED)
#define LAG_HALF    ((int)(MAX_LAG_S * DOA_FS) + 3)
#define LAG_LEN     (2 * LAG_HALF + 1)

/* Exponential averaging rate for the steered-response map. The effective
 * integration time is approximately 1/DOA_ALPHA frames.                     */
#define DOA_ALPHA   0.02f

/* Ring position to acquisition channel mapping. MIC_CH[m] gives the
 * interleaved buffer channel carrying the microphone mounted at ring angle
 * 2*pi*m/NMIC. An incorrect mapping rotates or reflects every reported
 * bearing by a fixed amount without affecting internal consistency.
 *
 * Acquisition rank order: IN2(PF9), IN3(PF7), IN4(PF5), IN6(PF10),
 * IN7(PF8), IN8(PF6), IN9(PF4), IN10(PC0).                                  */
static const uint8_t MIC_CH[NMIC] = {0, 1, 2, 3, 4, 5, 6, 7};

/* ===================== Working storage ================================== */

static float32_t s_window[NFFT];            /* Hann window coefficients      */
static float32_t s_x[NMIC][NFFT];           /* windowed time-domain frames   */
static float32_t s_X[NMIC][NFFT];           /* packed real-FFT outputs       */
static float32_t s_cross[NFFT];             /* cross-spectrum, one pair      */
static float32_t s_corr[NFFT];              /* inverse transform result      */
static float32_t s_pcorr[NPAIRS][LAG_LEN];  /* cropped pair correlations     */

static uint8_t   s_pi[NPAIRS], s_pj[NPAIRS];        /* pair index to (i,j)   */
static float32_t s_lag[DOA_NUM_ANGLES][NPAIRS];     /* steering delay table  */

static float32_t s_score[DOA_NUM_ANGLES];       /* per-frame response map    */
static float32_t s_score_avg[DOA_NUM_ANGLES];   /* accumulated response map  */
static int       s_score_init = 0;

static int s_k_lo, s_k_hi;                  /* inclusive band bin limits     */

static arm_rfft_fast_instance_f32 s_fft;

/* ===================== Instrumentation ================================== */
/* Written for external inspection only. Declared volatile so that the
 * computations producing them are not eliminated as unused.                 */

/* Steering table assertions, evaluated once during initialisation. */
volatile float dbg_max_lag     = 0.0f;  /* bound: (2R/c) * FS                */
volatile float dbg_pair0_at0   = 0.0f;  /* adjacent pair (0,1) at 0 deg      */
volatile float dbg_diam_at0    = 0.0f;  /* antipodal pair (0,4) at 0 deg     */
volatile float dbg_antisym_err = 0.0f;  /* full table; equals twice the skew */
volatile float dbg_geo_antisym = 0.0f;  /* geometry only; expected near zero */

/* Per-frame estimate, prior to temporal averaging. */
volatile float dbg_frame_az    = 0.0f;
volatile float dbg_frame_conf  = 0.0f;

/* Circular statistics over the most recent block of per-frame estimates.
 * dbg_cluster_R is the resultant length; for uniformly distributed estimates
 * it approaches 1/sqrt(DBG_STAT_N).                                         */
#define DBG_STAT_N  128

volatile float    dbg_cluster_R  = 0.0f;
volatile float    dbg_spread_deg = 0.0f;
volatile float    dbg_mean_az    = 0.0f;
volatile uint16_t dbg_hist[36];             /* completed 10-degree histogram */

static float    s_cs_x = 0.0f, s_cs_y = 0.0f;
static uint32_t s_cs_n = 0;
static uint16_t s_hist_acc[36];

/* ===================== Helpers ========================================== */

static inline float mic_phi(int m)
{
    return (2.0f * PI * (float)m) / (float)NMIC;
}

/* Propagation delay of microphone m relative to the array centre for a plane
 * wave arriving from azimuth theta. The sign convention makes the delay most
 * negative for a microphone facing the source.                              */
static inline float tau_mic(int m, float theta)
{
    return -(DOA_RADIUS / DOA_SOUND_SPEED) * cosf(theta - mic_phi(m));
}

/* Linear interpolation into a cropped correlation array. Index 0 corresponds
 * to a lag of -LAG_HALF samples.                                            */
static inline float corr_at(const float32_t *c, float lag)
{
    float x = lag + (float)LAG_HALF;

    if (x < 0.0f) x = 0.0f;
    if (x > (float)(LAG_LEN - 1)) x = (float)(LAG_LEN - 1);

    int   i = (int)x;
    float f = x - (float)i;

    if (i >= LAG_LEN - 1) return c[LAG_LEN - 1];
    return c[i] * (1.0f - f) + c[i + 1] * f;
}

/* ===================== Initialisation =================================== */

void DOA_Init(void)
{
    /* Hann window. Tapering the frame to zero at both ends suppresses the
     * spectral leakage introduced by the transform's implicit periodic
     * extension. Applied identically to all channels, it does not perturb
     * inter-channel phase.                                                  */
    for (int n = 0; n < NFFT; n++) {
        s_window[n] = 0.5f * (1.0f - cosf(2.0f * PI * (float)n / (float)(NFFT - 1)));
    }

    /* Enumerate the unordered microphone pairs. */
    int p = 0;
    for (int i = 0; i < NMIC; i++) {
        for (int j = i + 1; j < NMIC; j++) {
            s_pi[p] = (uint8_t)i;
            s_pj[p] = (uint8_t)j;
            p++;
        }
    }

    /* Band limits, converted from frequency to bin index at a spacing of
     * DOA_FS / NFFT hertz per bin.                                          */
    s_k_lo = (int)ceilf (DOA_F_LO * (float)NFFT / DOA_FS);
    s_k_hi = (int)floorf(DOA_F_HI * (float)NFFT / DOA_FS);
    if (s_k_lo < 1)         s_k_lo = 1;
    if (s_k_hi > NHALF - 1) s_k_hi = NHALF - 1;

    /* Steering delay table, with the acquisition skew folded in.
     *
     * The eight channels are converted sequentially rather than
     * simultaneously, so microphone i is sampled Delta_i after the sweep
     * begins and its sample n corresponds to continuous time nT + Delta_i:
     *
     *     x_i[n] = s(nT + Delta_i - tau_i) = s(nT - (tau_i - Delta_i))
     *
     * The effective delay of the discrete sequence is therefore
     * tau_i - Delta_i, and the pair correlation peaks at
     * (tau_i - tau_j) - (Delta_i - Delta_j). The skew difference is
     * subtracted from the geometric delay.                                  */
    for (int a = 0; a < DOA_NUM_ANGLES; a++) {
        float theta = (2.0f * PI * (float)a) / (float)DOA_NUM_ANGLES;

        for (int q = 0; q < NPAIRS; q++) {
            int i = s_pi[q], j = s_pj[q];

            float dtau_geo = tau_mic(i, theta) - tau_mic(j, theta);
            float skew     = ((float)MIC_CH[i] - (float)MIC_CH[j]) * DOA_SKEW_DT;

            s_lag[a][q] = (dtau_geo - skew) * DOA_FS;
        }
    }

    arm_rfft_fast_init_f32(&s_fft, NFFT);

    DOA_Reset();

    /* ---- Steering table assertions --------------------------------------
     * The following quantities have closed-form predicted values derived
     * from the array geometry, and are computed here for external
     * verification. No acoustic input is required.                          */

    dbg_max_lag = 0.0f;
    for (int a = 0; a < DOA_NUM_ANGLES; a++) {
        for (int q = 0; q < NPAIRS; q++) {
            float m = fabsf(s_lag[a][q]);
            if (m > dbg_max_lag) dbg_max_lag = m;
        }
    }

    dbg_pair0_at0 = s_lag[0][0];

    dbg_diam_at0 = 0.0f;
    for (int q = 0; q < NPAIRS; q++) {
        if (s_pi[q] == 0 && s_pj[q] == 4) {
            dbg_diam_at0 = s_lag[0][q];
            break;
        }
    }

    /* Antisymmetry of the complete table. The geometric term inverts under a
     * 180 degree bearing reversal while the skew term does not, so the
     * residual equals twice the skew.                                       */
    dbg_antisym_err = 0.0f;
    for (int q = 0; q < NPAIRS; q++) {
        float e = fabsf(s_lag[0][q] + s_lag[DOA_NUM_ANGLES / 2][q]);
        if (e > dbg_antisym_err) dbg_antisym_err = e;
    }

    /* Antisymmetry of the geometric term alone, which follows from
     * cos(theta + pi - phi) = -cos(theta - phi) and must vanish to within
     * floating-point rounding. Validates the propagation model independently
     * of the skew correction.                                               */
    dbg_geo_antisym = 0.0f;
    for (int q = 0; q < NPAIRS; q++) {
        int i = s_pi[q], j = s_pj[q];

        float g0   = tau_mic(i, 0.0f) - tau_mic(j, 0.0f);
        float g180 = tau_mic(i, PI)   - tau_mic(j, PI);

        float e = fabsf((g0 + g180) * DOA_FS);
        if (e > dbg_geo_antisym) dbg_geo_antisym = e;
    }
}

void DOA_Reset(void)
{
    s_score_init = 0;
}

/* ===================== Per-frame processing ============================= */

void DOA_Process(const uint16_t *interleaved, doa_result_t *out)
{
    /* ---- 1. De-interleave, remove DC, apply window ----------------------
     * The first pass is the only access to the acquisition buffer, which
     * resides in non-cacheable memory; the second operates on the extracted
     * copy. The DC level is measured per channel and per frame rather than
     * assumed, since each capsule sits at its own bias point and that point
     * drifts with temperature and supply.                                   */
    for (int c = 0; c < NMIC; c++) {
        float32_t *dst = s_x[c];
        float acc = 0.0f;

        for (int n = 0; n < NFFT; n++) {
            float v = (float)interleaved[n * NMIC + c];
            dst[n] = v;
            acc   += v;
        }

        float dc = acc * (1.0f / (float)NFFT);

        for (int n = 0; n < NFFT; n++) {
            dst[n] = (dst[n] - dc) * s_window[n];
        }
    }

    /* ---- 2. Forward transform ------------------------------------------ */
    for (int c = 0; c < NMIC; c++) {
        arm_rfft_fast_f32(&s_fft, s_x[c], s_X[c], 0);
    }

    /* ---- 3. PHAT cross-spectra and pair correlations --------------------
     * Packed real-FFT layout for length N:
     *   X[0] = DC, X[1] = Nyquist, X[2k], X[2k+1] = Re, Im of bin k
     *   for k = 1 .. N/2 - 1.
     *
     * Conjugate multiplication subtracts phases, so the cross-spectrum
     * carries the inter-channel phase difference with the source spectrum
     * cancelled. Normalising each bin to unit magnitude weights all retained
     * frequencies equally, which sharpens the correlation peak and improves
     * robustness to reverberation. Bins outside the configured band are set
     * to zero, since the normalisation would otherwise raise noise-only bins
     * to full weight.                                                       */
    for (int q = 0; q < NPAIRS; q++) {
        const float32_t *Xi = s_X[s_pi[q]];
        const float32_t *Xj = s_X[s_pj[q]];

        /* The transform overwrites its input, so the buffer is cleared on
         * every pair rather than once.                                      */
        memset(s_cross, 0, sizeof(s_cross));

        for (int k = s_k_lo; k <= s_k_hi; k++) {
            float re_i = Xi[2*k], im_i = Xi[2*k + 1];
            float re_j = Xj[2*k], im_j = Xj[2*k + 1];

            float cr = re_i * re_j + im_i * im_j;
            float ci = im_i * re_j - re_i * im_j;

            float inv = 1.0f / (sqrtf(cr * cr + ci * ci) + 1e-9f);

            s_cross[2*k]     = cr * inv;
            s_cross[2*k + 1] = ci * inv;
        }

        arm_rfft_fast_f32(&s_fft, s_cross, s_corr, 1);

        /* Retain only physically realisable lags. The transform output is
         * circular, so negative lags occupy the tail of the array.          */
        for (int L = -LAG_HALF; L <= LAG_HALF; L++) {
            int idx = (L >= 0) ? L : (NFFT + L);
            s_pcorr[q][L + LAG_HALF] = s_corr[idx];
        }
    }

    /* ---- 4. Steered-response summation ----------------------------------
     * Each candidate azimuth is scored by summing every pair correlation at
     * the lag that azimuth predicts. At the true bearing all pairs are
     * evaluated near their own maxima and the terms reinforce; elsewhere
     * each pair is evaluated at an arbitrary point and the terms largely
     * cancel.                                                               */
    float sum = 0.0f;
    for (int a = 0; a < DOA_NUM_ANGLES; a++) {
        float s = 0.0f;
        for (int q = 0; q < NPAIRS; q++) {
            s += corr_at(s_pcorr[q], s_lag[a][q]);
        }
        s_score[a] = s;
        sum += s;
    }

    /* Normalise to unit mean so that frames differing in absolute level
     * contribute equally to the accumulated map.                            */
    float mean = sum * (1.0f / (float)DOA_NUM_ANGLES);
    if (mean != 0.0f) {
        float inv_mean = 1.0f / mean;
        for (int a = 0; a < DOA_NUM_ANGLES; a++) s_score[a] *= inv_mean;
    }

    /* ---- Instrumentation: per-frame estimate and circular statistics ---- */
    {
        float fb = -1e30f;
        int   fa = 0;

        for (int a = 0; a < DOA_NUM_ANGLES; a++) {
            if (s_score[a] > fb) { fb = s_score[a]; fa = a; }
        }

        dbg_frame_az   = (float)fa * (360.0f / (float)DOA_NUM_ANGLES);
        dbg_frame_conf = fb;    /* map has unit mean, so this is peak/mean   */

        float r = dbg_frame_az * (PI / 180.0f);
        s_cs_x += cosf(r);
        s_cs_y += sinf(r);
        s_hist_acc[((int)(dbg_frame_az / 10.0f)) % 36]++;
        s_cs_n++;

        if (s_cs_n >= DBG_STAT_N) {
            float mx = s_cs_x / (float)s_cs_n;
            float my = s_cs_y / (float)s_cs_n;
            float R  = sqrtf(mx * mx + my * my);

            dbg_cluster_R = R;

            float a = atan2f(my, mx) * (180.0f / PI);
            if (a < 0.0f) a += 360.0f;
            dbg_mean_az = a;

            dbg_spread_deg = (R > 0.0f && R < 1.0f)
                           ? sqrtf(-2.0f * logf(R)) * (180.0f / PI)
                           : 0.0f;

            s_cs_x = 0.0f;
            s_cs_y = 0.0f;
            s_cs_n = 0;

            for (int b = 0; b < 36; b++) {
                dbg_hist[b]   = s_hist_acc[b];
                s_hist_acc[b] = 0;
            }
        }
    }

    /* ---- 5. Temporal integration ----------------------------------------
     * Averaging is applied to the response map rather than to the per-frame
     * bearing estimates. A frame in which the true bearing ranked second
     * still contributes support at the true bearing, whereas averaging
     * estimates would discard everything but each frame's maximum. It also
     * avoids the wrap discontinuity inherent in averaging angles.           */
    if (!s_score_init) {
        for (int a = 0; a < DOA_NUM_ANGLES; a++) s_score_avg[a] = s_score[a];
        s_score_init = 1;
    } else {
        for (int a = 0; a < DOA_NUM_ANGLES; a++) {
            s_score_avg[a] += DOA_ALPHA * (s_score[a] - s_score_avg[a]);
        }
    }

    /* ---- 6. Peak search and parabolic refinement ------------------------ */
    float best = -1e30f, avg_sum = 0.0f;
    int   best_a = 0;

    for (int a = 0; a < DOA_NUM_ANGLES; a++) {
        avg_sum += s_score_avg[a];
        if (s_score_avg[a] > best) { best = s_score_avg[a]; best_a = a; }
    }

    int am1 = (best_a - 1 + DOA_NUM_ANGLES) % DOA_NUM_ANGLES;
    int ap1 = (best_a + 1) % DOA_NUM_ANGLES;

    float ym1 = s_score_avg[am1];
    float y0  = s_score_avg[best_a];
    float yp1 = s_score_avg[ap1];

    float denom = ym1 - 2.0f * y0 + yp1;
    float delta = 0.0f;
    if (fabsf(denom) > 1e-12f) delta = 0.5f * (ym1 - yp1) / denom;

    /* On a nearly flat map the vertex may fall far outside the interval over
     * which the parabola was fitted.                                        */
    if (delta >  1.0f) delta =  1.0f;
    if (delta < -1.0f) delta = -1.0f;

    float az = ((float)best_a + delta) * (360.0f / (float)DOA_NUM_ANGLES);
    if (az < 0.0f)    az += 360.0f;
    if (az >= 360.0f) az -= 360.0f;

    /* ---- 7. Confidence --------------------------------------------------
     * Ratio of peak to mean over the accumulated map. A sharp isolated peak
     * indicates a single dominant source; a flat map indicates noise, no
     * source, or multiple sources of comparable strength.                   */
    float avg_mean = avg_sum * (1.0f / (float)DOA_NUM_ANGLES);
    float conf = (avg_mean != 0.0f) ? (best / avg_mean) : 0.0f;

    out->azimuth_deg = az;
    out->confidence  = conf;
    out->valid       = (conf > DOA_CONF_GATE) ? 1u : 0u;
}

/* ===================== Transform library self-test ====================== */

/* Transforms a sinusoid completing an integer number of cycles across the
 * analysis window, placing all energy in a single bin. A magnitude peak at
 * TEST_BIN confirms that the floating-point unit is active, that the
 * transform tables are linked, and that the library headers are consistent
 * with the compiled objects. None of these are detectable at build time.    */

#define TEST_N      256
#define TEST_BIN    8

static float32_t test_in[TEST_N];
static float32_t test_out[TEST_N];
static float32_t test_mag[TEST_N / 2];

volatile uint32_t  g_peak_idx;
volatile float32_t g_peak_val;

void doa_smoke_test(void)
{
    arm_rfft_fast_instance_f32 fft;

    for (int i = 0; i < TEST_N; i++) {
        test_in[i] = arm_sin_f32(2.0f * PI * TEST_BIN * i / TEST_N);
    }

    if (arm_rfft_fast_init_f32(&fft, TEST_N) != ARM_MATH_SUCCESS) {
        g_peak_idx = 0xFFFFFFFFu;   /* transform tables unavailable */
        return;
    }

    arm_rfft_fast_f32(&fft, test_in, test_out, 0);

    test_mag[0] = fabsf(test_out[0]);
    arm_cmplx_mag_f32(&test_out[2], &test_mag[1], TEST_N / 2 - 1);

    arm_max_f32(test_mag, TEST_N / 2,
                (float32_t *)&g_peak_val, (uint32_t *)&g_peak_idx);
}
