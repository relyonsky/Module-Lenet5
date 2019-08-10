
/*
 *
 * www.osrc.cn
 * www.milinker.com
 * copyright by nan jin mi lian dian zi www.osrc.cn
 * axi dma test
 *
*/


#include "dma_intr.h"
#include "timer_intr.h"
#include "sys_intr.h"
#include "xgpio.h"

#include "lwip/err.h"
#include "lwip/tcp.h"
#include "lwipopts.h"
#include "netif/xadapter.h"
#include "lwipopts.h"
#include "tcp_transmission.h"

static  XScuGic Intc; //GIC
static  XScuTimer Timer;//timer
XAxiDma AxiDma;
u16 	*RxBufferPtr[2];  /* ping pong buffers*/

volatile u32 RX_success;
volatile u32 TX_success;

volatile u32 RX_ready=1;
volatile u32 TX_ready=1;

#define TIMER_LOAD_VALUE    XPAR_CPU_CORTEXA9_0_CPU_CLK_FREQ_HZ / 8 //0.25S


char oled_str[17]="";
static XGpio Gpio;

#define AXI_GPIO_DEV_ID	        XPAR_AXI_GPIO_0_DEVICE_ID

int init_intr_sys(void)
{
	DMA_Intr_Init(&AxiDma,0);//initial interrupt system
	Timer_init(&Timer,TIMER_LOAD_VALUE,0);
	Init_Intr_System(&Intc); // initial DMA interrupt system
	Setup_Intr_Exception(&Intc);
	DMA_Setup_Intr_System(&Intc,&AxiDma,0,RX_INTR_ID);//setup dma interrpt system
	Timer_Setup_Intr_System(&Intc,&Timer,TIMER_IRPT_INTR);
	DMA_Intr_Enable(&Intc,&AxiDma);

}

int main(void)
{

	int Status;
	struct netif *netif, server_netif;
	struct ip_addr ipaddr, netmask, gw;
	err_t err;

	/* the mac address of the board. this should be unique per board */
	unsigned char mac_ethernet_address[] = { 0x00, 0x0a, 0x35, 0x00, 0x01, 0x02 };

	/* Initialize the ping pong buffers for the data received from axidma */
	RxBufferPtr[0] = (u16 *)RX_BUFFER0_BASE;
	RxBufferPtr[1] = (u16 *)RX_BUFFER1_BASE;

	XGpio_Initialize(&Gpio, AXI_GPIO_DEV_ID);
	XGpio_SetDataDirection(&Gpio, 1, 0);
	init_intr_sys();
	TcpTmrFlag = 0;

	netif = &server_netif;

	IP4_ADDR(&ipaddr,  192, 168,   1,  10);
	IP4_ADDR(&netmask, 255, 255, 255,  0);
	IP4_ADDR(&gw,      192, 168,   10,  1);

	/*lwip library init*/
	lwip_init();
	/* Add network interface to the netif_list, and set it as default */
	netif_set_default(netif);
	/* specify that the network if is up */
	netif_set_up(netif);
	/* initialize tcp pcb */
	tcp_send_init();

	XGpio_DiscreteWrite(&Gpio, 1, 1);
	Timer_start(&Timer);


	while (1)
	{
		/* call tcp timer every 250ms */
		if(TcpTmrFlag)
		{
			tcp_tmr();
			TcpTmrFlag = 0;
		}
		/*receive input packet from emac*/
		xemacif_input(netif);//transfer packets from MAC queue to the LwIP/IP stack
		/* if connected to the server, start receive data from PL through axidma, then transmit the data to the PC software by TCP*/
		if(tcp_client_connected)
			send_received_data();
	}
	return 0;

}


