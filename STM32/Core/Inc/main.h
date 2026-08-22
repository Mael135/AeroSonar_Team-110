/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Common definitions for the acquisition layer.
  *
  * Declares the peripheral handles, the acquisition buffer and the
  * instrumentation counters shared between main.c and the interrupt
  * handlers in stm32h7xx_it.c.
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

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32h7xx_hal.h"

#include "stm32h7xx_nucleo.h"
#include <stdio.h>

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "doa.h"
/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* Peripheral handles, defined in main.c. The interrupt handlers in
 * stm32h7xx_it.c dispatch through these.                                    */
extern ADC_HandleTypeDef hadc3;
extern DMA_HandleTypeDef hdma_adc3;
extern TIM_HandleTypeDef htim6;

/* Frame availability flag, set by the DMA completion handlers and cleared by
 * the main loop. 0: none pending, 1: first half ready, 2: second half ready.
 * Declared volatile as it is modified in interrupt context.                 */
extern volatile int doa_half_ready;

/* Most recent bearing estimate. */
extern doa_result_t doa_res;

/* Instrumentation, defined in main.c.
 *   doa_frames    frames processed since reset
 *   doa_overruns  frames arriving before the previous one completed;
 *                 expected to remain zero in steady state
 *   doa_cycles    processor cycles consumed by the most recent call to
 *                 DOA_Process, excluding the channel health survey
 *   ch_dc, ch_pp  per-channel DC level and peak-to-peak amplitude over the
 *                 most recent frame, in converter counts                    */
extern volatile uint32_t doa_frames;
extern volatile uint32_t doa_overruns;
extern volatile uint32_t doa_cycles;
extern volatile uint16_t ch_dc[8];
extern volatile uint16_t ch_pp[8];

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
