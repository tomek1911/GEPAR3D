// assets/js/vtk-viewer.js

function initVTKViewer(container, vtpFile, scalarType = 'discrete', windowViewWidth = 0.25) {
  // Set container size
  const width = window.innerWidth * windowViewWidth;
  container.style.width = `${width}px`;
  container.style.height = `${width}px`;

  const grw = vtk.Rendering.Misc.vtkGenericRenderWindow.newInstance({
    background: [0.8, 0.8, 0.8],
    listenWindowResize: true
  });
  grw.setContainer(container);
  grw.resize();

  const renderer = grw.getRenderer();
  const renderWindow = grw.getRenderWindow();

  const actor = vtk.Rendering.Core.vtkActor.newInstance();
  renderer.addActor(actor);

  // Initial camera setup
  const camera = renderer.getActiveCamera();
  camera.setViewUp(0, 0, 1);
  camera.setPosition(0, 1, 0);

  // Helper: convert hex to [r,g,b,a] in 0–1
  function hexToRgb01(hex, alpha = 1.0) {
    const bigint = parseInt(hex.slice(1), 16);
    const r = ((bigint >> 16) & 255) / 255.0;
    const g = ((bigint >> 8) & 255) / 255.0;
    const b = (bigint & 255) / 255.0;
    return [r, g, b, alpha];
  }

  async function loadAsset(vtpFile, scalarTypeOverride) {
    const type = scalarTypeOverride || scalarType;
    const reader = vtk.IO.XML.vtkXMLPolyDataReader.newInstance();

    try {
      // Wait until VTP file is loaded
      await reader.setUrl(vtpFile);
      await reader.loadData();

      const polydata = reader.getOutputData();
      const mapper = vtk.Rendering.Core.vtkMapper.newInstance();
      mapper.setInputData(polydata);

      const scalars = polydata.getPointData().getScalars();

      if (scalars) {
        const lut = vtk.Rendering.Core.vtkColorTransferFunction.newInstance();

        if (type === 'discrete') {
          // Wait until colormap is fetched and parsed
          const text = await fetch('assets/data/colormap.txt').then(res => res.text());
          const lines = text.trim().split('\n');
          lines.forEach(line => {
            const [row, cls, r, g, b, a] = line.trim().split(/\s+/).map(Number);
            lut.addRGBPoint(cls, r / 255, g / 255, b / 255);
          });

          mapper.setLookupTable(lut);
          mapper.setColorModeToMapScalars();
          mapper.setScalarModeToUsePointData();
          mapper.setScalarVisibility(true);
          mapper.setScalarRange(0, 32);
        } else if (type === 'continuous') {
          lut.setNanColor(0.5, 0.5, 0.5, 1.0);
          lut.setBelowRangeColor(hexToRgb01("#82cc12"));
          lut.setAboveRangeColor(hexToRgb01("#c05eeb"));
          lut.setUseBelowRangeColor(true);
          lut.setUseAboveRangeColor(true);

          const cvals = [1.5, 2, 3, 4, 5, 5.5];
          const colors = ["#a1cc12", "#faf561", "#fadf69", "#fa9e61", "#f75757", "#c05eeb"];
          cvals.forEach((val, i) => {
            const [r, g, b, a] = hexToRgb01(colors[i]);
            lut.addRGBPoint(val, r, g, b);
          });

          mapper.setLookupTable(lut);
          mapper.setColorModeToMapScalars();
          mapper.setScalarModeToUsePointData();
          mapper.setScalarVisibility(true);
          mapper.setScalarRange(0.4, 2.0);
        }
      }

      actor.setMapper(mapper);

      // Actor appearance
      actor.getProperty().setAmbient(0.2);
      actor.getProperty().setDiffuse(0.7);
      actor.getProperty().setSpecular(0.3);
      actor.getProperty().setSpecularPower(10.0);

      renderer.resetCamera();
      renderWindow.render();

    } catch (err) {
      console.error('Failed to load VTP or colormap:', err);
    }
  }

  // Initial load
  loadAsset(vtpFile);

  // Return object for external control
  return { actor, renderer, renderWindow, loadAsset };
}
