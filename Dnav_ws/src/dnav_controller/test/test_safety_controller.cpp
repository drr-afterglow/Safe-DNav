#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include <dnav_controller/safety_controller.h>

namespace dc = dnav_controller;

namespace {

dc::ControllerInput activeInput() {
  dc::ControllerInput input;
  input.navigation_enabled = true;
  input.odometry_valid = true;
  input.attitude_valid = true;
  input.flight_state_valid = true;
  input.flight_state_flying = true;
  input.policy_valid = true;
  input.altitude = 2.0;
  input.dt = 1.0;
  return input;
}

dc::SafetyConfig unlimitedAccelerationConfig() {
  dc::SafetyConfig config;
  config.max_horizontal_acceleration = 100.0;
  config.max_vertical_acceleration = 100.0;
  return config;
}

}  // namespace

TEST(SafetyController, StatePriorityAndLimiterReset) {
  dc::SafetyController controller(dc::SafetyConfig{});
  auto input = activeInput();
  input.estop_latched = true;
  input.odometry_valid = false;
  EXPECT_EQ(controller.update(input).state, dc::ControllerState::ESTOP);

  input.estop_latched = false;
  EXPECT_EQ(controller.update(input).state, dc::ControllerState::WAITING_FOR_ODOMETRY);
  input.odometry_valid = true;
  input.attitude_valid = false;
  EXPECT_EQ(controller.update(input).state, dc::ControllerState::FAULT);
  input.attitude_valid = true;
  input.flight_state_valid = false;
  EXPECT_EQ(controller.update(input).state, dc::ControllerState::WAITING_FOR_FLIGHT_STATE);
  input.flight_state_valid = true;
  input.flight_state_flying = false;
  EXPECT_EQ(controller.update(input).state, dc::ControllerState::DISABLED);
  input.flight_state_flying = true;
  input.navigation_enabled = false;
  EXPECT_EQ(controller.update(input).state, dc::ControllerState::DISABLED);
  input.navigation_enabled = true;
  input.policy_valid = false;
  EXPECT_EQ(controller.update(input).state, dc::ControllerState::WAITING_FOR_POLICY);
}

TEST(SafetyController, HorizontalSpeedUsesVectorNorm) {
  dc::SafetyController controller(unlimitedAccelerationConfig());
  auto input = activeInput();
  input.policy_velocity_world = {3.0, 4.0, 0.0};
  const auto output = controller.update(input);
  EXPECT_EQ(output.state, dc::ControllerState::ACTIVE);
  EXPECT_TRUE(output.horizontal_speed_limited);
  EXPECT_NEAR(output.limited_velocity_world.x, 0.3, 1e-12);
  EXPECT_NEAR(output.limited_velocity_world.y, 0.4, 1e-12);
  EXPECT_NEAR(std::hypot(output.limited_velocity_world.x,
                         output.limited_velocity_world.y), 0.5, 1e-12);
}

TEST(SafetyController, VerticalAndAltitudeLimits) {
  dc::SafetyController controller(unlimitedAccelerationConfig());
  auto input = activeInput();
  input.policy_velocity_world.z = -3.0;
  input.altitude = 0.75;
  auto output = controller.update(input);
  EXPECT_TRUE(output.vertical_speed_limited);
  EXPECT_TRUE(output.altitude_limited);
  EXPECT_NEAR(output.limited_velocity_world.z, -0.15, 1e-12);

  controller.resetLimiter();
  input.policy_velocity_world.z = 3.0;
  input.altitude = 3.75;
  output = controller.update(input);
  EXPECT_TRUE(output.vertical_speed_limited);
  EXPECT_TRUE(output.altitude_limited);
  EXPECT_NEAR(output.limited_velocity_world.z, 0.2, 1e-12);

  controller.resetLimiter();
  input.policy_velocity_world.z = -0.2;
  input.altitude = 0.4;
  output = controller.update(input);
  EXPECT_NEAR(output.limited_velocity_world.z, 0.0, 1e-12);
}

TEST(SafetyController, AccelerationLimitStartsFromZero) {
  dc::SafetyConfig config;
  dc::SafetyController controller(config);
  auto input = activeInput();
  input.dt = 0.1;
  input.policy_velocity_world = {0.5, 0.0, 0.4};
  auto output = controller.update(input);
  EXPECT_TRUE(output.acceleration_limited);
  EXPECT_NEAR(output.limited_velocity_world.x, 0.08, 1e-12);
  EXPECT_NEAR(output.limited_velocity_world.z, 0.05, 1e-12);
  output = controller.update(input);
  EXPECT_NEAR(output.limited_velocity_world.x, 0.16, 1e-12);
  EXPECT_NEAR(output.limited_velocity_world.z, 0.10, 1e-12);

  input.navigation_enabled = false;
  EXPECT_EQ(controller.update(input).state, dc::ControllerState::DISABLED);
  input.navigation_enabled = true;
  output = controller.update(input);
  EXPECT_NEAR(output.limited_velocity_world.x, 0.08, 1e-12);
}

TEST(SafetyController, YawConversionMatchesHeadingFrame) {
  dc::SafetyController controller(unlimitedAccelerationConfig());
  auto input = activeInput();
  input.yaw = M_PI_2;
  input.policy_velocity_world = {0.4, -0.2, 0.1};
  const auto output = controller.update(input);
  EXPECT_NEAR(output.command_velocity_heading.x, -0.2, 1e-12);
  EXPECT_NEAR(output.command_velocity_heading.y, -0.4, 1e-12);
  EXPECT_NEAR(output.command_velocity_heading.z, 0.1, 1e-12);
}

TEST(SafetyController, RejectsNonFiniteInputAndBadPeriod) {
  dc::SafetyController controller(dc::SafetyConfig{});
  auto input = activeInput();
  input.policy_velocity_world.x = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(controller.update(input).state, dc::ControllerState::FAULT);
  input.policy_velocity_world.x = 0.0;
  input.dt = 0.0;
  EXPECT_EQ(controller.update(input).state, dc::ControllerState::FAULT);
}

TEST(SafetyController, RejectsInvalidConfiguration) {
  dc::SafetyConfig config;
  config.min_altitude = config.max_altitude;
  EXPECT_THROW(dc::SafetyController controller(config), std::invalid_argument);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
