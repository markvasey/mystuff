package com.example.housepricemonitor;

import com.example.housepricemonitor.controller.DashboardController;
import com.example.housepricemonitor.service.HousePriceAnalyticsService;
import com.example.housepricemonitor.service.HousePricePoller;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.view;

@WebMvcTest(DashboardController.class)
public class HousePriceMonitorApplicationTests {

	@Autowired
	private MockMvc mockMvc;

	@MockitoBean
	private HousePriceAnalyticsService analyticsService;

	@MockitoBean
	private HousePricePoller poller;

	@Test
	public void testDashboardAccess() throws Exception {
		mockMvc.perform(get("/dashboard"))
				.andExpect(status().isOk())
				.andExpect(view().name("dashboard"));
	}

	@Test
	public void testDashboardCaseInsensitiveAccess() throws Exception {
		mockMvc.perform(get("/DASHBOARD"))
				.andExpect(status().isOk())
				.andExpect(view().name("dashboard"));
	}

	@Test
	public void testPollAccess() throws Exception {
		mockMvc.perform(get("/poll"))
				.andExpect(status().is3xxRedirection());
	}
}
