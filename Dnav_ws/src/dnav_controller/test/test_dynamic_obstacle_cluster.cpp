#include <gtest/gtest.h>

#include <dnav_controller/dynamic_obstacle_cluster.h>

namespace dc = dnav_controller;

namespace {
dc::DynamicGridSample sample(double distance, double radial, double sigma,
                             double z = 0.0, double absolute_speed = 0.5) {
  return {{distance, 0.0, z, radial, sigma, true}, distance, absolute_speed, true};
}
}

TEST(DynamicObstacleCluster, RejectsIsolatedPixel) {
  dc::DynamicClusterConfig config;
  std::vector<dc::DynamicGridSample> grid(config.height*config.width);
  grid[10] = sample(2.0, -0.5, 0.2);
  EXPECT_TRUE(dc::clusterDynamicObstacles(grid, config).empty());
}

TEST(DynamicObstacleCluster, AggregatesConnectedObjectWithMedians) {
  dc::DynamicClusterConfig config;
  std::vector<dc::DynamicGridSample> grid(config.height*config.width);
  grid[2*config.width+8] = sample(2.0, -1.0, 0.1);
  grid[2*config.width+9] = sample(2.1, 0.2, 0.3);
  grid[3*config.width+8] = sample(2.2, -0.4, 0.2);
  const auto result = dc::clusterDynamicObstacles(grid, config);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_NEAR(result[0].x, 2.1, 1e-12);
  EXPECT_NEAR(result[0].relative_radial_velocity, -0.4, 1e-12);
  EXPECT_NEAR(result[0].radial_velocity_uncertainty, 0.2, 1e-12);
}

TEST(DynamicObstacleCluster, SeparatesDepthDiscontinuityAndWrapsAzimuth) {
  dc::DynamicClusterConfig config;
  std::vector<dc::DynamicGridSample> grid(config.height*config.width);
  grid[config.width] = sample(1.5, -0.2, 0.1);
  grid[2*config.width-1] = sample(1.6, -0.2, 0.1);  // wraps to column zero
  grid[10] = sample(3.0, -0.3, 0.2);
  grid[11] = sample(3.1, -0.3, 0.2);
  const auto result = dc::clusterDynamicObstacles(grid, config);
  ASSERT_EQ(result.size(), 2u);
  EXPECT_NEAR(result[0].x, 1.55, 1e-12);
  EXPECT_NEAR(result[1].x, 3.05, 1e-12);
}

TEST(DynamicObstacleCluster, RejectsInvalidConfiguration) {
  dc::DynamicClusterConfig config;
  config.minimum_pixels = 0;
  EXPECT_THROW(dc::clusterDynamicObstacles({}, config), std::invalid_argument);
}

TEST(DynamicObstacleCluster, RejectsStaticAndImplausiblyFastClusters) {
  dc::DynamicClusterConfig config;
  std::vector<dc::DynamicGridSample> grid(config.height*config.width);
  grid[8] = sample(2.0, -0.2, 0.1, 0.0, 0.05);
  grid[9] = sample(2.1, -0.2, 0.1, 0.0, 0.05);
  grid[18] = sample(2.5, -0.2, 0.1, 0.0, 4.0);
  grid[19] = sample(2.6, -0.2, 0.1, 0.0, 4.0);
  EXPECT_TRUE(dc::clusterDynamicObstacles(grid, config).empty());
}

TEST(DynamicObstacleCluster, MergesSplitPartsOfOneObjectConservatively) {
  dc::DynamicClusterConfig config;
  std::vector<dc::DynamicGridSample> grid(config.height*config.width);
  grid[2*config.width+4] = sample(2.0, -0.2, 0.1);
  grid[3*config.width+4] = sample(2.1, -0.2, 0.1);
  grid[2*config.width+7] = sample(2.6, -0.7, 0.4);
  grid[3*config.width+7] = sample(2.7, -0.7, 0.4);
  const auto result = dc::clusterDynamicObstacles(grid, config);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_NEAR(result[0].x, 2.05, 1e-12);
  EXPECT_NEAR(result[0].relative_radial_velocity, -0.7, 1e-12);
  EXPECT_NEAR(result[0].radial_velocity_uncertainty, 0.4, 1e-12);
}

TEST(DynamicObstacleTracker, HoldsAndPredictsAcrossShortDropout) {
  dc::DynamicObstacleTracker tracker(dc::DynamicTrackConfig{});
  dc::ObstaclePoint approaching{2.0, 0.0, 0.0, -0.5, 0.1, true};
  tracker.update({approaching}, 1.0);
  tracker.update({}, 1.2);
  auto result = tracker.obstacles(1.4);
  ASSERT_EQ(result.size(), 1u);
  EXPECT_NEAR(result[0].x, 1.8, 1e-12);
  EXPECT_TRUE(tracker.obstacles(1.61).empty());
}

TEST(DynamicObstacleTracker, RejectsFarJumpWithoutDroppingExistingTrack) {
  dc::DynamicObstacleTracker tracker(dc::DynamicTrackConfig{});
  tracker.update({{2.0, 0.0, 0.0, -0.5, 0.1, true}}, 1.0);
  tracker.update({{5.0, 0.0, 0.0, 0.0, 0.1, true}}, 1.1);
  const auto result = tracker.obstacles(1.1);
  ASSERT_EQ(result.size(), 2u);
  EXPECT_LT(result[0].x, 2.0);
  EXPECT_NEAR(result[1].x, 5.0, 1e-12);
}

TEST(DynamicObstacleTracker, SmoothsMatchedDetectionInsteadOfJumping) {
  dc::DynamicTrackConfig config;
  config.update_alpha = 0.25;
  dc::DynamicObstacleTracker tracker(config);
  tracker.update({{2.0, 0.0, 0.0, -0.4, 0.2, true}}, 1.0);
  tracker.update({{2.4, 0.0, 0.0, 0.4, 0.6, true}}, 1.1);
  const auto result = tracker.obstacles(1.1);
  ASSERT_EQ(result.size(), 1u);
  // The old point first propagates to 1.96 m, then receives a 25% update.
  EXPECT_NEAR(result[0].x, 2.07, 1e-12);
  EXPECT_NEAR(result[0].relative_radial_velocity, -0.2, 1e-12);
  EXPECT_NEAR(result[0].radial_velocity_uncertainty, 0.3, 1e-12);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
