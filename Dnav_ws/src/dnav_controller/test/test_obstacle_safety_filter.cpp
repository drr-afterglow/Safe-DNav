#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <dnav_controller/obstacle_safety_filter.h>

namespace dc = dnav_controller;

TEST(ObstacleSafetyFilter, LeavesSafeOrTangentialVelocityUnchanged) {
  dc::ObstacleSafetyFilter filter(dc::ObstacleFilterConfig{});
  const std::vector<dc::ObstaclePoint> obstacles{{3.0, 0.0, 0.0}};
  auto result = filter.filter({0.5, 0.0, 0.1}, {0.5, 0.0, 0.0}, obstacles);
  EXPECT_FALSE(result.active);
  EXPECT_NEAR(result.safe_velocity_body.x, 0.5, 1e-12);
  result = filter.filter({0.0, 0.5, 0.1}, {0.5, 0.0, 0.0}, obstacles);
  EXPECT_FALSE(result.active);
  EXPECT_NEAR(result.safe_velocity_body.y, 0.5, 1e-12);
}

TEST(ObstacleSafetyFilter, LimitsClosingSpeedUsingMeasuredBrakingModel) {
  dc::ObstacleFilterConfig config;
  config.safety_margin = 0.20;
  dc::ObstacleSafetyFilter filter(config);
  const std::vector<dc::ObstaclePoint> obstacles{{0.7, 0.0, 0.0}};
  const auto result = filter.filter({0.45, 0.2, 0.1}, {0.1, 0.0, 0.0}, obstacles);
  const double expected = -0.25*0.15 +
      std::sqrt(std::pow(0.25*0.15, 2) + 2.0*0.25*0.2);
  EXPECT_TRUE(result.active);
  EXPECT_EQ(result.active_constraints, 1u);
  EXPECT_NEAR(result.safe_velocity_body.x, expected, 1e-12);
  EXPECT_NEAR(result.safe_velocity_body.y, 0.2, 1e-12);
  EXPECT_NEAR(result.safe_velocity_body.z, 0.1, 1e-12);
}

TEST(ObstacleSafetyFilter, AppliesDeploymentSpeedEnvelopeBeforeObstacleConstraints) {
  dc::ObstacleSafetyFilter filter(dc::ObstacleFilterConfig{});
  const auto result = filter.filter({3.0, 4.0, 0.1}, {}, {});
  EXPECT_FALSE(result.active);
  EXPECT_NEAR(result.safe_velocity_body.x, 0.3, 1e-12);
  EXPECT_NEAR(result.safe_velocity_body.y, 0.4, 1e-12);
  EXPECT_NEAR(result.safe_velocity_body.z, 0.1, 1e-12);
}

TEST(ObstacleSafetyFilter, CommandsImmediateRadialStopWhenAlreadyInsideBrakingEnvelope) {
  dc::ObstacleSafetyFilter filter(dc::ObstacleFilterConfig{});
  const auto result = filter.filter({0.45, 0.2, 0.0}, {0.5, 0.0, 0.0},
                                    {{1.0, 0.0, 0.0}});
  EXPECT_TRUE(result.active);
  EXPECT_NEAR(result.safe_velocity_body.x, 0.0, 1e-12);
  EXPECT_NEAR(result.safe_velocity_body.y, 0.2, 1e-12);
}

TEST(ObstacleSafetyFilter, AccountsForApproachingDynamicObstacle) {
  dc::ObstacleSafetyFilter filter(dc::ObstacleFilterConfig{});
  dc::ObstaclePoint moving{2.0, 0.0, 0.0, -1.0, 0.0, true};
  const auto result = filter.filter({0.5, 0.0, 0.0}, {0.5, 0.0, 0.0}, {moving});
  // dr/dt=-1 with ego closing at 0.5 means the obstacle itself approaches at
  // 0.5 m/s. At this range the measured relative stopping envelope is already
  // exceeded, so the command must retreat at that same speed.
  EXPECT_TRUE(result.active);
  EXPECT_EQ(result.dynamic_constraints, 1u);
  EXPECT_NEAR(result.safe_velocity_body.x, -0.5, 1e-12);
}

TEST(ObstacleSafetyFilter, UsesDynamicVelocityUncertaintyConservatively) {
  dc::ObstacleFilterConfig config;
  config.dynamic_confidence_scale = 2.0;
  dc::ObstacleSafetyFilter filter(config);
  dc::ObstaclePoint moving{1.0, 0.0, 0.0, 0.0, 0.1, true};
  const auto result = filter.filter({0.5, 0.0, 0.0}, {}, {moving});
  EXPECT_EQ(result.dynamic_constraints, 1u);
  EXPECT_NEAR(result.safe_velocity_body.x, 0.0, 1e-12);
}

TEST(ObstacleSafetyFilter, CapsImpossibleDynamicRetreatAtSpeedEnvelope) {
  dc::ObstacleSafetyFilter filter(dc::ObstacleFilterConfig{});
  dc::ObstaclePoint moving{2.0, 0.0, 0.0, -3.0, 1.0, true};
  const auto result = filter.filter({0.5, 0.0, 0.0}, {}, {moving});
  EXPECT_TRUE(result.active);
  EXPECT_NEAR(result.safe_velocity_body.x, -0.5, 1e-12);
}

TEST(ObstacleSafetyFilter, ORCAUsesTangentialMotionForCrossingObstacle) {
  dc::ObstacleSafetyFilter filter(dc::ObstacleFilterConfig{});
  dc::ObstaclePoint crossing{0.7, 0.2, 0.0, 0.0, 0.0, true, -0.5};
  const auto result = filter.filter({0.5, 0.0, 0.0}, {0.5, 0.0, 0.0},
                                    {crossing});
  EXPECT_TRUE(result.active);
  EXPECT_EQ(result.dynamic_constraints, 1u);
  EXPECT_LT(result.safe_velocity_body.x, 0.5);
  EXPECT_NE(std::abs(result.safe_velocity_body.y), 0.0);
  EXPECT_LE(std::hypot(result.safe_velocity_body.x, result.safe_velocity_body.y),
            0.5 + 1e-9);
}

TEST(ObstacleSafetyFilter, BiasesEscapeAwayFromCloseMoverSide) {
  dc::ObstacleSafetyFilter filter(dc::ObstacleFilterConfig{});
  dc::ObstaclePoint crossing{1.0, -0.25, 0.0, 0.0, 0.0, true, 0.5};
  const auto result = filter.filter({0.5, 0.0, 0.0}, {}, {crossing},
                                    {5.0, 0.0, 0.0});
  EXPECT_TRUE(result.active);
  EXPECT_GT(result.safe_velocity_body.y, 0.0);
}

TEST(ObstacleSafetyFilter, StaticMisclassificationMatchesStaticConstraintWithoutInflation) {
  dc::ObstacleFilterConfig config;
  config.dynamic_confidence_scale = 0.0;
  config.dynamic_safety_margin = config.safety_margin;
  dc::ObstacleSafetyFilter filter(config);
  const dc::Vector3 actual{0.3, 0.0, 0.0};
  const auto static_result = filter.filter({0.5, 0.0, 0.0}, actual,
                                           {{1.5, 0.0, 0.0}});
  dc::ObstaclePoint mislabeled{1.5, 0.0, 0.0, -0.3, 2.5, true};
  const auto dynamic_result = filter.filter({0.5, 0.0, 0.0}, actual, {mislabeled});
  EXPECT_NEAR(dynamic_result.safe_velocity_body.x,
              static_result.safe_velocity_body.x, 1e-12);
}

TEST(ObstacleSafetyFilter, CapsUncertaintyVelocityContribution) {
  dc::ObstacleFilterConfig config;
  config.dynamic_confidence_scale = 2.0;
  config.dynamic_uncertainty_velocity_cap = 0.3;
  dc::ObstacleSafetyFilter filter(config);
  dc::ObstaclePoint noisy{1.5, 0.0, 0.0, 0.0, 10.0, true};
  const auto result = filter.filter({0.5, 0.0, 0.0}, {}, {noisy});
  const double available =
      1.5-config.vehicle_radius-config.dynamic_safety_margin;
  const double allowed = -config.effective_deceleration*config.latency_budget+
      std::sqrt(std::pow(config.effective_deceleration*config.latency_budget, 2)+
                2.0*config.effective_deceleration*available);
  EXPECT_NEAR(result.safe_velocity_body.x, allowed, 1e-12);
  EXPECT_GT(result.safe_velocity_body.x, 0.0);
}

TEST(ObstacleSafetyFilter, UsesLargerDynamicSafetyDistance) {
  dc::ObstacleFilterConfig config;
  dc::ObstacleSafetyFilter filter(config);
  const dc::ObstaclePoint static_point{1.0, 0.0, 0.0};
  const dc::ObstaclePoint dynamic_point{1.0, 0.0, 0.0, 0.0, 0.0, true};
  const auto static_result = filter.filter({0.2, 0.0, 0.0}, {}, {static_point});
  const auto dynamic_result = filter.filter({0.2, 0.0, 0.0}, {}, {dynamic_point});
  const double available = std::max(
      0.0, 1.0-config.vehicle_radius-config.dynamic_safety_margin);
  const double expected = -config.effective_deceleration*config.latency_budget+
      std::sqrt(std::pow(config.effective_deceleration*config.latency_budget, 2)+
                2.0*config.effective_deceleration*available);
  EXPECT_FALSE(static_result.active);
  EXPECT_TRUE(dynamic_result.active);
  EXPECT_LT(dynamic_result.safe_velocity_body.x,
            static_result.safe_velocity_body.x);
  EXPECT_NEAR(dynamic_result.safe_velocity_body.x, expected, 1e-12);
}

TEST(ObstacleSafetyFilter, StopsClosingInsideSafetyMarginAndIgnoresGround) {
  dc::ObstacleSafetyFilter filter(dc::ObstacleFilterConfig{});
  auto result = filter.filter({0.5, 0.0, 0.0}, {0.5, 0.0, 0.0},
                              {{0.45, 0.0, 0.0}, {0.2, 0.0, -0.8}});
  EXPECT_TRUE(result.active);
  EXPECT_NEAR(result.safe_velocity_body.x, 0.0, 1e-12);
  EXPECT_NEAR(result.minimum_surface_clearance, 0.15, 1e-12);
}

TEST(ObstacleSafetyFilter, SimultaneouslySatisfiesAllConstraintsRegardlessOfOrder) {
  dc::ObstacleSafetyFilter filter(dc::ObstacleFilterConfig{});
  const double diagonal = 0.45/std::sqrt(2.0);
  const std::vector<dc::ObstaclePoint> first_order{
      {0.45, 0.0, 0.0}, {-diagonal, diagonal, 0.0}};
  const std::vector<dc::ObstaclePoint> reverse_order{
      {-diagonal, diagonal, 0.0}, {0.45, 0.0, 0.0}};
  const auto first = filter.filter({0.4, 0.3, 0.0}, {}, first_order);
  const auto reverse = filter.filter({0.4, 0.3, 0.0}, {}, reverse_order);

  EXPECT_TRUE(first.active);
  EXPECT_LE(first.safe_velocity_body.x, 1e-10);
  EXPECT_LE(-first.safe_velocity_body.x+first.safe_velocity_body.y, 1e-10);
  EXPECT_NEAR(first.safe_velocity_body.x, reverse.safe_velocity_body.x, 1e-10);
  EXPECT_NEAR(first.safe_velocity_body.y, reverse.safe_velocity_body.y, 1e-10);
}

TEST(ObstacleSafetyFilter, UsesGoalDirectedFeasibleVelocityToEscapeStopDeadlock) {
  dc::ObstacleSafetyFilter filter(dc::ObstacleFilterConfig{});
  const auto result = filter.filter({0.5, 0.0, 0.0}, {},
                                    {{0.45, 0.0, 0.0}}, {1.0, 1.0, 0.0});
  EXPECT_TRUE(result.active);
  EXPECT_TRUE(result.escape_assist);
  EXPECT_LE(result.safe_velocity_body.x, 1e-10);
  EXPECT_GT(result.safe_velocity_body.y, 0.45);
  EXPECT_LE(std::hypot(result.safe_velocity_body.x, result.safe_velocity_body.y),
            0.5+1e-10);
}

TEST(ObstacleSafetyFilter, RestoresGoalProgressWhenPolicyMeanCollapsesToZero) {
  dc::ObstacleSafetyFilter filter(dc::ObstacleFilterConfig{});
  const auto open = filter.filter({}, {}, {}, {3.0, 4.0, 0.0});
  EXPECT_TRUE(open.active);
  EXPECT_TRUE(open.escape_assist);
  EXPECT_NEAR(open.safe_velocity_body.x, 0.3, 1e-10);
  EXPECT_NEAR(open.safe_velocity_body.y, 0.4, 1e-10);

  const auto blocked = filter.filter({}, {}, {{0.45, 0.0, 0.0}},
                                     {1.0, 1.0, 0.0});
  EXPECT_TRUE(blocked.escape_assist);
  EXPECT_LE(blocked.safe_velocity_body.x, 1e-10);
  EXPECT_GT(blocked.safe_velocity_body.y, 0.45);
}

TEST(ObstacleSafetyFilter, UsesLeastViolatingEscapeInsteadOfStoppingWhenInfeasible) {
  dc::ObstacleFilterConfig config;
  config.dynamic_confidence_scale = 0.0;
  dc::ObstacleSafetyFilter filter(config);
  // An approaching mover in front requires negative x, while a close wall
  // behind requires non-negative x. No command satisfies both constraints.
  dc::ObstaclePoint mover{0.9, 0.0, 0.0, -0.5, 0.0, true};
  dc::ObstaclePoint wall{-0.4, 0.0, 0.0};
  const auto result = filter.filter({0.5, 0.0, 0.0}, {}, {mover, wall});
  EXPECT_TRUE(result.active);
  EXPECT_TRUE(result.escape_assist);
  EXPECT_LT(result.safe_velocity_body.x, 0.0);
  EXPECT_GT(std::abs(result.safe_velocity_body.y), 0.0);
  EXPECT_NEAR(std::hypot(result.safe_velocity_body.x,
                         result.safe_velocity_body.y),
              config.max_horizontal_speed, 1e-12);
}

TEST(ObstacleSafetyFilter, RejectsInvalidConfigurationAndVelocity) {
  dc::ObstacleFilterConfig config;
  config.effective_deceleration = 0.0;
  EXPECT_THROW(dc::ObstacleSafetyFilter filter(config), std::invalid_argument);
  dc::ObstacleSafetyFilter filter(dc::ObstacleFilterConfig{});
  EXPECT_THROW(filter.filter({std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0},
                             {}, {}), std::invalid_argument);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
