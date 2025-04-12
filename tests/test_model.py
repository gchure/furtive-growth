import pytest
import numpy as np
from growth.model import Species

class TestSpecies:
    """Test suite for the Species class."""

    def test_initialization_default_values(self):
        """Test that Species initializes with correct default values."""
        species = Species()
        
        # Test core parameters
        assert species.lambda_max == 1.0
        assert species.Km == 0.01
        assert species.gamma == 0.1
        assert species.Y == 0.1
        
        # Test state variables
        assert species.growth_rate == 0.0
        assert species.effective_growth_rate == 0.0
        assert species.extinct is False
        assert species.fixed is False

    def test_initialization_custom_values(self):
        """Test that Species initializes with custom parameter values."""
        species = Species(lambda_max=2.0, Km=0.05, gamma=0.2, Y=0.15)
        
        assert species.lambda_max == 2.0
        assert species.Km == 0.05
        assert species.gamma == 0.2
        assert species.Y == 0.15

    def test_update_zero_nutrient(self):
        """Test update method with zero nutrient concentration."""
        species = Species(lambda_max=2.0, Km=0.5)
        species.update(0.0)
        
        assert species.growth_rate == 0.0
        assert species.effective_growth_rate == -species.gamma
    
    def test_update_high_nutrient(self):
        """Test update method with high nutrient concentration (saturation)."""
        species = Species(lambda_max=2.0, Km=0.5)

        # Using very high concentration to test saturation
        species.update(1_000_000)
        
        # At high concentration, growth rate should approach lambda_max
        assert np.isclose(species.growth_rate, species.lambda_max, rtol=1e-3)
        assert np.isclose(species.effective_growth_rate, species.lambda_max - species.gamma)
    
    def test_update_half_saturation(self):
        """Test update method at half-saturation point (c_nt = Km)."""
        species = Species(lambda_max=2.0, Km=0.5)
        species.update(species.Km)
        
        # At c_nt = Km, growth rate should be lambda_max/2
        assert np.isclose(species.growth_rate, species.lambda_max/2)
        assert np.isclose(species.effective_growth_rate, species.lambda_max/2 - species.gamma)
    
    def test_update_negative_nutrient(self):
        """Test update method with negative nutrient concentration (should be treated as 0)."""
        species = Species()
        species.update(-1.0)
        
        assert species.growth_rate == 0.0
        assert species.effective_growth_rate == -species.gamma
    
    def test_compute_derivatives_zero_biomass(self):
        """Test compute_derivatives with zero biomass."""
        species = Species()
        derivatives = species.compute_derivatives(M=0.0, c_nt=1.0, delta=0.1)
        
        assert derivatives[0] == 0.0  # dM_dt should be zero
        assert derivatives[1] == 0.0  # dc_nt_dt should be zero
    
    def test_compute_derivatives_zero_nutrient(self):
        """Test compute_derivatives with zero nutrient concentration."""
        species = Species()
        derivatives = species.compute_derivatives(M=1.0, c_nt=0.0, delta=0.1)
        
        assert derivatives[0] == -0.2  # dM_dt = M * (-gamma - delta) = 1 * (-0.1 - 0.1)
        assert derivatives[1] == 0.0   # dc_nt_dt should be zero since growth_rate is zero
    
    def test_compute_derivatives_normal_conditions(self):
        """Test compute_derivatives under normal growth conditions."""
        species = Species(lambda_max=1.0, Km=0.5, gamma=0.1, Y=0.2)
        
        # Calculate expected values manually
        c_nt = 2.0
        M = 1.0
        delta = 0.05
        
        # Expected growth_rate = lambda_max * c_nt / (Km + c_nt)
        expected_growth_rate = 1.0 * 2.0 / (0.5 + 2.0)
        # Expected effective_growth_rate = growth_rate - gamma
        expected_effective_growth_rate = expected_growth_rate - 0.1
        # Expected dM_dt = M * (effective_growth_rate - delta)
        expected_dM_dt = M * (expected_effective_growth_rate - delta)
        # Expected dc_nt_dt = -growth_rate * M / Y
        expected_dc_nt_dt = -expected_growth_rate * M / 0.2
        
        derivatives = species.compute_derivatives(M=M, c_nt=c_nt, delta=delta)
        
        assert np.isclose(derivatives[0], expected_dM_dt)
        assert np.isclose(derivatives[1], expected_dc_nt_dt)
        
        # Also verify that the species state was updated
        assert np.isclose(species.growth_rate, expected_growth_rate)
        assert np.isclose(species.effective_growth_rate, expected_effective_growth_rate)
    
    def test_get_data(self):
        """Test get_data method returns the correct dictionary with species state."""
        species = Species(lambda_max=2.0, Km=0.5, gamma=0.2, Y=0.3)
        species.update(1.0)  # Update with some nutrient concentration
        
        data = species.get_data()
        
        # Check that all required keys are present
        expected_keys = {
            'growth_rate', 'growth_rate_max', 'growth_rate_eff', 
            'gamma', 'Km', 'Y', 'extinct', 'fixed'
        }
        assert set(data.keys()) == expected_keys
        
        # Check values
        assert data['growth_rate'] == species.growth_rate
        assert data['growth_rate_max'] == species.lambda_max
        assert data['growth_rate_eff'] == species.effective_growth_rate
        assert data['gamma'] == species.gamma
        assert data['Km'] == species.Km
        assert data['Y'] == species.Y
        assert data['extinct'] == species.extinct
        assert data['fixed'] == species.fixed
    
    def test_monod_equation_correctness(self):
        """Test that growth follows the Monod equation across a range of nutrient concentrations."""
        species = Species(lambda_max=2.0, Km=0.5)
        
        # Test across a range of nutrient concentrations
        nutrient_concentrations = np.logspace(-3, 2, 100)  # From 0.001 to 100
        growth_rates = []
        
        for c_nt in nutrient_concentrations:
            species.update(c_nt)
            growth_rates.append(species.growth_rate)
        
        # Calculate expected growth rates from Monod equation
        expected_rates = [2.0 * c / (0.5 + c) for c in nutrient_concentrations]
        
        # Check that all calculated rates match expected
        assert np.allclose(growth_rates, expected_rates)
    
    def test_positive_parameter_validation(self):
        """Test that all parameters must be positive."""
        # These should be valid
        Species(lambda_max=0.1, Km=0.01, gamma=0, Y=0.1)  # gamma can be zero
    
        # These should raise ValueError  
        with pytest.raises(ValueError, match="lambda_max must be non-negative"):
            Species(lambda_max=-1.0)
    
        with pytest.raises(ValueError, match="Km must be positive"):
            Species(Km=0)
    
        with pytest.raises(ValueError, match="Km must be positive"):
            Species(Km=-0.5)
    
        with pytest.raises(ValueError, match="gamma must be non-negative"):
            Species(gamma=-0.1)
    
        with pytest.raises(ValueError, match="Y must be positive"):
            Species(Y=0)
    
        with pytest.raises(ValueError, match="Y must be positive"):
            Species(Y=-0.2)

    def test_scientific_notation_parameters(self):
        """Test that parameters can be specified using scientific notation."""
        species = Species(lambda_max=1e-3, Km=1e-5, gamma=1e-4, Y=1e-2)

        assert species.lambda_max == 1e-3
        assert species.Km == 1e-5
        assert species.gamma == 1e-4
        assert species.Y == 1e-2

        def test_negative_parameters_warning(self):
            """Test that negative parameter values raise a warning."""
            # In a proper Monod model, these parameters shouldn't be negative
            # Depending on your implementation, you might want to add validation
            with pytest.warns(UserWarning):
                species = Species(lambda_max=-1.0)  # This assumes you add a warning for negative values

        def test_effective_growth_rate_calculation(self):
            """Test that effective_growth_rate is correctly calculated as growth_rate - gamma."""
            species = Species(lambda_max=1.5, gamma=0.3)

            # Test with different nutrient concentrations
            for c_nt in [0.0, 0.1, 1.0, 10.0]:
                species.update(c_nt)
                assert species.effective_growth_rate == species.growth_rate - species.gamma