document.addEventListener("DOMContentLoaded", () => {
  const viewers = [];

  const viewerContainers = [
    "vtk-container-1",
    "vtk-container-2",
    "vtk-container-3",
    "vtk-container-4",
    "vtk-container-5",
    "vtk-container-6"
  ];

  const assets = {
    normal: {
      A: [
        "assets/data/VTK_data/GEPAR3D/A.vtp",
        "assets/data/VTK_data/SGANET/A.vtp",
        "assets/data/VTK_data/TOOTHSEG/A.vtp"
      ],
      B: [
        "assets/data/VTK_data/GEPAR3D/B.vtp",
        "assets/data/VTK_data/SGANET/B.vtp",
        "assets/data/VTK_data/TOOTHSEG/B.vtp"
      ]
    },
    error: {
      A: [
        "assets/data/VTK_data/error_maps/GEPAR3D/A.vtp",
        "assets/data/VTK_data/error_maps/SGANET/A.vtp",
        "assets/data/VTK_data/error_maps/TOOTHSEG/A.vtp"
      ],
      B: [
        "assets/data/VTK_data/error_maps/GEPAR3D/B.vtp",
        "assets/data/VTK_data/error_maps/SGANET/B.vtp",
        "assets/data/VTK_data/error_maps/TOOTHSEG/B.vtp"
      ]
    }
  };

  // Initialize viewers with scan A
  viewerContainers.forEach((id, i) => {
    const container = document.getElementById(id);
    const scanType = i < 3 ? "normal" : "error";
    const idx = i % 3;
    const scalarType = scanType === "normal" ? "discrete" : "continuous";
    const viewer = initVTKViewer(container, assets[scanType].A[idx], scalarType);
    viewers.push(viewer);
  });

  // Generate bullet slider
  const assetSetNames = ["CenterA", "CenterB"];
  const sliderContainer = document.getElementById("assetSetSlider");
  const loadingIndicator = document.getElementById("loadingIndicator"); // make sure you added this in HTML

  assetSetNames.forEach((name, index) => {
    const bullet = document.createElement("span");
    bullet.classList.add("bullet");
    if (index === 0) bullet.classList.add("active");
    bullet.dataset.index = index;
    bullet.title = name;
    sliderContainer.appendChild(bullet);

    bullet.addEventListener("click", async () => {
      // Show loading
      loadingIndicator.style.display = "block";

      // Update active bullet
      sliderContainer.querySelectorAll(".bullet").forEach(b => b.classList.remove("active"));
      bullet.classList.add("active");

      const scan = index === 0 ? "A" : "B";
      document.getElementById("scanLabel").textContent = scan;

      // Load all assets in parallel
      const loadPromises = viewers.map((viewer, i) => {
        const type = i < 3 ? "normal" : "error";
        const idx = i % 3;
        const scalarType = type === "normal" ? "discrete" : "continuous";
        return viewer.loadAsset(assets[type][scan][idx], scalarType);
      });

      try {
        await Promise.all(loadPromises); // wait for all to finish
      } catch (err) {
        console.error("Error loading assets:", err);
      }

      // Hide loading
      loadingIndicator.style.display = "none";
    });
  });
});
