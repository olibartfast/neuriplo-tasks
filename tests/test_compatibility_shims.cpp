#include "neuriplo/tasks/core/bounding_box.hpp"
#include "vision-core/core/bounding_box.hpp"

#include <gtest/gtest.h>
#include <type_traits>

TEST(CompatibilityShims, OldNamespaceAliasesNewNamespace) {
    static_assert(std::is_same_v<vision_core::BoundingBox, neuriplo_tasks::BoundingBox>);

    vision_core::BoundingBox box(1, 2, 3, 4);
    EXPECT_EQ(box.area(), 12);
}
