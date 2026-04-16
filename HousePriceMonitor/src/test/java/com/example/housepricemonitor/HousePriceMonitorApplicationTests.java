package com.example.housepricemonitor;

import com.example.housepricemonitor.controller.DashboardController;
import com.example.housepricemonitor.service.ComparisonConfigService;
import com.example.housepricemonitor.service.HousePriceAnalyticsService;
import com.example.housepricemonitor.service.HousePricePoller;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import java.util.Collections;

import static org.mockito.Mockito.when;
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

	@MockitoBean
	private ComparisonConfigService comparisonConfigService;

	@Test
	public void testDashboardAccess() throws Exception {
		when(analyticsService.getTransactionsByDistrict()).thenReturn(Collections.emptyMap());
		when(comparisonConfigService.getAllCriteria()).thenReturn(Collections.emptyMap());

		mockMvc.perform(get("/dashboard"))
				.andExpect(status().isOk())
				.andExpect(view().name("dashboard"));
	}

	@Test
	public void testDashboardCaseInsensitiveAccess() throws Exception {
		when(analyticsService.getTransactionsByDistrict()).thenReturn(Collections.emptyMap());
		when(comparisonConfigService.getAllCriteria()).thenReturn(Collections.emptyMap());

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
