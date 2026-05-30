package com.example.jobsearch.service;

import com.example.jobsearch.client.JobSourceClient;
import com.example.jobsearch.repository.JobListingRepository;
import com.example.jobsearch.repository.SearchCriteriaRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;

import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class JobPollerServiceTest {

    @Mock
    private SearchCriteriaRepository criteriaRepository;

    @Mock
    private JobListingRepository jobListingRepository;

    @Mock
    private RelevanceScorerService relevanceScorerService;

    @Mock
    private JobSourceClient mockClient;

    private JobPollerService jobPollerService;

    @BeforeEach
    void setUp() {
        jobPollerService = new JobPollerService(
                criteriaRepository,
                jobListingRepository,
                List.of(mockClient),
                relevanceScorerService
        );
    }

    @Test
    void testPollJobs_WhenDisabled() {
        // Set cronExpression to "-"
        ReflectionTestUtils.setField(jobPollerService, "cronExpression", "-");

        jobPollerService.pollJobs();

        // Verify that the repository was never called (polling skipped)
        verify(criteriaRepository, never()).findByActiveTrue();
    }

    @Test
    void testPollJobs_WhenEnabled() {
        // Set cronExpression to a valid cron string
        ReflectionTestUtils.setField(jobPollerService, "cronExpression", "0 0 * * * *");
        
        when(criteriaRepository.findByActiveTrue()).thenReturn(List.of());

        jobPollerService.pollJobs();

        // Verify that the repository was called (polling proceeded)
        verify(criteriaRepository, times(1)).findByActiveTrue();
    }

    @Test
    void testPollJobs_PreventsReentry() {
        ReflectionTestUtils.setField(jobPollerService, "cronExpression", "0 0 * * * *");
        
        // Simulate a long running poll by making the first call to repo take time or just manually setting the flag
        ReflectionTestUtils.setField(jobPollerService, "polling", new java.util.concurrent.atomic.AtomicBoolean(true));

        jobPollerService.pollJobs();

        // Verify that the repository was NOT called because polling was already "true"
        verify(criteriaRepository, never()).findByActiveTrue();
    }
}
