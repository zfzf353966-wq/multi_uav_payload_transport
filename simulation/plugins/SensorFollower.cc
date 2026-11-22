// Update virtual sensor mounts inside the simulator, without transport latency.
#include <gz/plugin/Register.hh>
#include <gz/sim/System.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/Pose.hh>
#include <string>

namespace mupt
{
class SensorFollower final : public gz::sim::System,
                             public gz::sim::ISystemPreUpdate
{
  public: void PreUpdate(const gz::sim::UpdateInfo &info,
                        gz::sim::EntityComponentManager &ecm) override
  {
    if (info.paused) return;
    for (int index = 1; index <= 3; ++index)
    {
      auto vehicle = ecm.EntityByComponents(gz::sim::components::Model(),
          gz::sim::components::Name("x500_" + std::to_string(index)));
      auto sensor = ecm.EntityByComponents(gz::sim::components::Model(),
          gz::sim::components::Name("uav_" + std::to_string(index) + "_lidar"));
      if (vehicle == gz::sim::kNullEntity || sensor == gz::sim::kNullEntity) continue;
      auto pose = gz::sim::worldPose(vehicle, ecm);
      pose.Pos().Z() += 0.60;
      pose.Rot() = gz::math::Quaterniond(0, 0, pose.Rot().Yaw());
      // Pods are static root models; updating Pose directly also avoids the
      // one-step physics-command delay. Physics/collisions of UAVs are untouched.
      ecm.SetComponentData<gz::sim::components::Pose>(sensor, pose);
    }
  }
};
}
GZ_ADD_PLUGIN(mupt::SensorFollower, gz::sim::System, gz::sim::ISystemPreUpdate)
