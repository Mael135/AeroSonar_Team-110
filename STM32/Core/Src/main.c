/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Acquisition and processing entry point for the acoustic
  *                   localization subsystem.
  *
  * Eight microphone channels are sampled by a single converter under timer
  * pacing and transferred to memory by DMA without processor involvement.
  * The processor is interrupted twice per buffer traversal and performs all
  * signal processing from the main loop.
  *
  * Signal chain:
  *   TIM6 (25 kHz TRGO) -> ADC3 (8-rank scan, 16-bit) -> BDMA (circular)
  *   -> adc_buf in SRAM4 -> DOA_Process -> bearing estimate
  *
  * The acquisition buffer must reside in the D3 domain, since BDMA is the
  * only controller able to serve ADC3 and can master only D3 memory. Its
  * placement is fixed by an explicit linker section; see MPU_Config for the
  * corresponding cache attributes.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "doa.h"
#include <stdio.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

COM_InitTypeDef BspCOMInit;
ADC_HandleTypeDef hadc3;
DMA_HandleTypeDef hdma_adc3;

TIM_HandleTypeDef htim6;

/* USER CODE BEGIN PV */

/* Acquisition geometry. ADC_CH is the converter sequence length; ADC_FRAMES
 * is the frame length in samples per channel. The trailing factor of two
 * provides the two half-buffers: the DMA controller fills one while the
 * processor operates on the other.                                          */
#define ADC_CH       8
#define ADC_FRAMES   1024
#define ADC_BUF_LEN  (ADC_CH * ADC_FRAMES * 2)

/* A mismatch between the acquisition geometry and the estimator's frame
 * configuration produces no diagnostic at run time: the transform would
 * silently process the wrong number of samples.                             */
_Static_assert(ADC_FRAMES == DOA_N,
               "Frame length must match DOA_N");
_Static_assert(ADC_CH == DOA_NUM_MIC,
               "Converter sequence length must match DOA_NUM_MIC");

/* Placed in SRAM4 by the .RAM_D3 linker section. KEEP is applied there
 * because the buffer is referenced only through the DMA controller and would
 * otherwise be eligible for removal by link-time garbage collection.        */
__attribute__((section(".RAM_D3")))
uint16_t adc_buf[ADC_BUF_LEN];

/* Set by the DMA completion handlers, cleared by the main loop.
 * 0: no frame pending, 1: first half ready, 2: second half ready.           */
volatile int doa_half_ready = 0;

doa_result_t doa_res;

/* Instrumentation. doa_cycles measures DOA_Process alone; the channel health
 * survey is deliberately excluded from the timing window. doa_overruns
 * counts frames that arrived before the previous one had been processed and
 * should remain at zero in steady state.                                    */
volatile uint32_t doa_frames   = 0;
volatile uint32_t doa_overruns = 0;
volatile uint32_t doa_cycles   = 0;

/* Per-channel DC level and peak-to-peak amplitude over the current frame,
 * in converter counts.                                                      */
volatile uint16_t ch_dc[ADC_CH];
volatile uint16_t ch_pp[ADC_CH];

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MPU_Config(void);
static void MX_GPIO_Init(void);
static void MX_BDMA_Init(void);
static void MX_ADC3_Init(void);
static void MX_TIM6_Init(void);
/* USER CODE BEGIN PFP */
void PeriphCommonClock_Config(void);
static void channel_health(const uint16_t *frame);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/**
  * @brief  Record per-channel DC level and peak-to-peak amplitude.
  * @param  frame  One half-buffer, ADC_FRAMES * ADC_CH samples in rank order.
  *
  * Used to confirm that every channel is populated, correctly biased and
  * acoustically responsive. Channel c is extracted with stride ADC_CH.
  */
static void channel_health(const uint16_t *frame)
{
    for (int c = 0; c < ADC_CH; c++) {
        uint16_t mn = 65535, mx = 0;
        uint32_t acc = 0;

        for (int n = 0; n < ADC_FRAMES; n++) {
            uint16_t v = frame[n * ADC_CH + c];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            acc += v;
        }

        ch_dc[c] = (uint16_t)(acc / ADC_FRAMES);
        ch_pp[c] = mx - mn;
    }
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MPU Configuration--------------------------------------------------------*/
  MPU_Config();

  /* Enable the CPU Cache */

  /* Enable I-Cache---------------------------------------------------------*/
  SCB_EnableICache();

  /* Enable D-Cache---------------------------------------------------------*/
  SCB_EnableDCache();

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */
  PeriphCommonClock_Config();
  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_BDMA_Init();
  MX_ADC3_Init();
  MX_TIM6_Init();

  /* USER CODE BEGIN 2 */

  /* Enable the cycle counter, used to measure per-frame processing cost. */
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  DWT->CYCCNT = 0;
  DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;

  DOA_Init();

  /* The converter and DMA controller are armed before the pacing timer is
   * started, so that the first trigger is not issued into an inactive
   * transfer path.                                                          */
  if (HAL_ADC_Start_DMA(&hadc3, (uint32_t *)adc_buf, ADC_BUF_LEN) != HAL_OK)
  {
      Error_Handler();
  }
  HAL_TIM_Base_Start(&htim6);

  /* USER CODE END 2 */

  /* Initialize COM1 port (115200, 8 bits (7-bit data + 1 stop bit), no parity */
  BspCOMInit.BaudRate   = 115200;
  BspCOMInit.WordLength = COM_WORDLENGTH_8B;
  BspCOMInit.StopBits   = COM_STOPBITS_1;
  BspCOMInit.Parity     = COM_PARITY_NONE;
  BspCOMInit.HwFlowCtl  = COM_HWCONTROL_NONE;
  if (BSP_COM_Init(COM1, &BspCOMInit) != BSP_ERROR_NONE)
  {
    Error_Handler();
  }

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    int half = doa_half_ready;

    if (half != 0)
    {
      /* Cleared before processing rather than after, so that a frame
       * arriving mid-computation is preserved and detected below.           */
      doa_half_ready = 0;

      const uint16_t *frame = (half == 1)
                            ? &adc_buf[0]
                            : &adc_buf[ADC_CH * ADC_FRAMES];

      channel_health(frame);

      uint32_t t0 = DWT->CYCCNT;
      DOA_Process(frame, &doa_res);
      doa_cycles = DWT->CYCCNT - t0;

      doa_frames++;

      /* A flag already set on completion indicates that the frame period
       * was exceeded and samples have been lost.                            */
      if (doa_half_ready != 0) doa_overruns++;
    }
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  *
  * HSE bypass at 8 MHz from the on-board debug interface drives PLL1 to a
  * 400 MHz system clock (M=1, N=100, P=2). The AHB divider of 4 places the
  * bus domains at 100 MHz, which sets the timer input clock and hence the
  * sample rate. FLASH_LATENCY_1 is required at 100 MHz HCLK under voltage
  * scale 1.
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Supply configuration update enable
  */
  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 1;
  RCC_OscInitStruct.PLL.PLLN = 100;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_3;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_D3PCLK1|RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV1;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief ADC3 Initialization Function
  * @param None
  * @retval None
  *
  * Eight-rank regular sequence at 16-bit resolution, triggered externally by
  * TIM6 and delivering results to DMA in circular mode. Continuous and
  * discontinuous conversion are both disabled, so one trigger initiates
  * exactly one complete traversal of the sequence and the per-channel sample
  * rate equals the trigger rate.
  *
  * Rank order defines the buffer interleave: rank r occupies index r within
  * each group of eight. A sampling time of 32.5 cycles is required for
  * 16-bit settling from the 2.2 kohm source impedance of the microphone
  * bias network; incomplete settling would transfer residual charge between
  * consecutive ranks, corrupting the inter-channel phase relationships from
  * which bearing is derived.
  */
static void MX_ADC3_Init(void)
{

  /* USER CODE BEGIN ADC3_Init 0 */

  /* USER CODE END ADC3_Init 0 */

  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC3_Init 1 */

  /* USER CODE END ADC3_Init 1 */

  /** Common config
  */
  hadc3.Instance = ADC3;
  hadc3.Init.ClockPrescaler = ADC_CLOCK_ASYNC_DIV1;
  hadc3.Init.ScanConvMode = ADC_SCAN_ENABLE;
  hadc3.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  hadc3.Init.LowPowerAutoWait = DISABLE;
  hadc3.Init.ContinuousConvMode = DISABLE;
  hadc3.Init.NbrOfConversion = 8;
  hadc3.Init.DiscontinuousConvMode = DISABLE;
  hadc3.Init.ExternalTrigConv = ADC_EXTERNALTRIG_T6_TRGO;
  hadc3.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_RISING;
  hadc3.Init.ConversionDataManagement = ADC_CONVERSIONDATA_DMA_CIRCULAR;
  hadc3.Init.Overrun = ADC_OVR_DATA_OVERWRITTEN;
  hadc3.Init.LeftBitShift = ADC_LEFTBITSHIFT_NONE;
  hadc3.Init.OversamplingMode = DISABLE;
  hadc3.Init.Oversampling.Ratio = 1;
  if (HAL_ADC_Init(&hadc3) != HAL_OK)
  {
    Error_Handler();
  }
  hadc3.Init.Resolution = ADC_RESOLUTION_16B;
  if (HAL_ADC_Init(&hadc3) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_2;
  sConfig.Rank = ADC_REGULAR_RANK_1;
  sConfig.SamplingTime = ADC_SAMPLETIME_32CYCLES_5;
  sConfig.SingleDiff = ADC_SINGLE_ENDED;
  sConfig.OffsetNumber = ADC_OFFSET_NONE;
  sConfig.Offset = 0;
  sConfig.OffsetSignedSaturation = DISABLE;
  if (HAL_ADC_ConfigChannel(&hadc3, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_3;
  sConfig.Rank = ADC_REGULAR_RANK_2;
  if (HAL_ADC_ConfigChannel(&hadc3, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_4;
  sConfig.Rank = ADC_REGULAR_RANK_3;
  if (HAL_ADC_ConfigChannel(&hadc3, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_6;
  sConfig.Rank = ADC_REGULAR_RANK_4;
  if (HAL_ADC_ConfigChannel(&hadc3, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_7;
  sConfig.Rank = ADC_REGULAR_RANK_5;
  if (HAL_ADC_ConfigChannel(&hadc3, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_8;
  sConfig.Rank = ADC_REGULAR_RANK_6;
  if (HAL_ADC_ConfigChannel(&hadc3, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_9;
  sConfig.Rank = ADC_REGULAR_RANK_7;
  if (HAL_ADC_ConfigChannel(&hadc3, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_10;
  sConfig.Rank = ADC_REGULAR_RANK_8;
  if (HAL_ADC_ConfigChannel(&hadc3, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC3_Init 2 */

  /* USER CODE END ADC3_Init 2 */

}

/**
  * @brief TIM6 Initialization Function
  * @param None
  * @retval None
  *
  * Provides the sample clock. With a 100 MHz input, prescaler 0 and an
  * auto-reload of 3999, the update event occurs every 4000 ticks, giving
  * 25 kHz. The event is routed internally to the trigger bus as TRGO; the
  * timer drives no external pin. Timing is therefore established entirely in
  * hardware and is unaffected by interrupt latency or processor load, which
  * matters because the measurand is inter-channel timing.
  */
static void MX_TIM6_Init(void)
{

  /* USER CODE BEGIN TIM6_Init 0 */

  /* USER CODE END TIM6_Init 0 */

  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM6_Init 1 */

  /* USER CODE END TIM6_Init 1 */
  htim6.Instance = TIM6;
  htim6.Init.Prescaler = 0;
  htim6.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim6.Init.Period = 3999;
  htim6.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;
  if (HAL_TIM_Base_Init(&htim6) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_UPDATE;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim6, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM6_Init 2 */

  /* USER CODE END TIM6_Init 2 */

}

/**
  * @brief  Enable DMA controller clock and interrupt.
  *
  * The channel itself (request source, direction, transfer width, circular
  * mode) is configured in HAL_ADC_MspInit.
  */
static void MX_BDMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_BDMA_CLK_ENABLE();

  /* DMA interrupt init */
  /* BDMA_Channel0_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(BDMA_Channel0_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(BDMA_Channel0_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  *
  * Analog mode for the converter inputs is configured in HAL_ADC_MspInit.
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOF_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_7, GPIO_PIN_RESET);

  /*Configure GPIO pin : PB7 */
  GPIO_InitStruct.Pin = GPIO_PIN_7;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/**
  * @brief  Configure the converter kernel clock.
  *
  * The converter is clocked independently of the bus domains. PLL2 divides
  * the 8 MHz reference by 2 and multiplies by 100 to a 400 MHz oscillator,
  * from which P=8 yields a 50 MHz kernel clock. With ADC_CLOCK_ASYNC_DIV1
  * this is the conversion clock, and it determines the inter-rank interval
  * corrected by DOA_SKEW_DT.
  */
void PeriphCommonClock_Config(void)
{
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};

  PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_ADC;
  PeriphClkInitStruct.PLL2.PLL2M = 2;
  PeriphClkInitStruct.PLL2.PLL2N = 100;
  PeriphClkInitStruct.PLL2.PLL2P = 8;
  PeriphClkInitStruct.PLL2.PLL2Q = 2;
  PeriphClkInitStruct.PLL2.PLL2R = 2;
  PeriphClkInitStruct.PLL2.PLL2RGE = RCC_PLL2VCIRANGE_3;
  PeriphClkInitStruct.PLL2.PLL2VCOSEL = RCC_PLL2VCOWIDE;
  PeriphClkInitStruct.PLL2.PLL2FRACN = 0;
  PeriphClkInitStruct.AdcClockSelection = RCC_ADCCLKSOURCE_PLL2;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
}

/* DMA completion handlers. These signal frame availability and perform no
 * processing: the transform occupies a substantial fraction of the frame
 * period and executing it here would delay subsequent interrupts and
 * eventually cause sample loss.                                             */

void HAL_ADC_ConvHalfCpltCallback(ADC_HandleTypeDef *hadc)
{
    if (hadc->Instance == ADC3) doa_half_ready = 1;
}

void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc)
{
    if (hadc->Instance == ADC3) doa_half_ready = 2;
}

/**
  * @brief  Retarget standard output to the instrumentation trace port.
  *
  * Requires trace output to be enabled in the debug configuration; with it
  * disabled the characters are discarded.
  */
int _write(int file, char *ptr, int len)
{
    (void)file;

    for (int i = 0; i < len; i++) {
        ITM_SendChar((uint32_t)(uint8_t)ptr[i]);
    }

    return len;
}

/* USER CODE END 4 */

/**
  * @brief  Configure the memory protection unit.
  *
  * Marks the acquisition buffer region non-cacheable. The processor's
  * write-back data cache is not coherent with DMA writes, so a cached read
  * of the buffer could return stale data. Defining the attribute here avoids
  * issuing explicit cache maintenance around every access. The region is
  * also marked execute-never, as it holds data exclusively.
  *
  * Must run before the caches are enabled.
  */
static void MPU_Config(void)
{
  MPU_Region_InitTypeDef MPU_InitStruct = {0};

  /* Disables the MPU */
  HAL_MPU_Disable();

  /** Initializes and configures the Region and the memory to be protected
  */
  MPU_InitStruct.Enable = MPU_REGION_ENABLE;
  MPU_InitStruct.Number = MPU_REGION_NUMBER0;
  MPU_InitStruct.BaseAddress = 0x38000000;      /* SRAM4, D3 domain */
  MPU_InitStruct.Size = MPU_REGION_SIZE_64KB;
  MPU_InitStruct.SubRegionDisable = 0x00;
  MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL0;
  MPU_InitStruct.AccessPermission = MPU_REGION_FULL_ACCESS;
  MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
  MPU_InitStruct.IsShareable = MPU_ACCESS_SHAREABLE;
  MPU_InitStruct.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
  MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;

  HAL_MPU_ConfigRegion(&MPU_InitStruct);

  /* PRIVDEF retains the default memory map as a background region, so that
   * addresses not covered by an explicit region remain accessible.          */
  HAL_MPU_Enable(MPU_HFNMI_PRIVDEF);
}

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
