// assets/js/vtk-viewer.js

function initVTKViewer(containerId, vtpFile, scalarType='discrete', windowViewWidth = 0.25) {
  window.requestAnimationFrame(() => {
    const container = document.querySelector(containerId);

    // Set container size
    const width = window.innerWidth * windowViewWidth;
    container.style.width = `${width}px`;
    container.style.height = `${width}px`;

    const grw = vtk.Rendering.Misc.vtkGenericRenderWindow.newInstance({
      background: [0.8, 0.8, 0.8],
      listenWindowResize: true,
    });
    grw.setContainer(container);
    grw.resize();

    const renderer = grw.getRenderer();
    const renderWindow = grw.getRenderWindow();

    const reader = vtk.IO.XML.vtkXMLPolyDataReader.newInstance();

    fetch(vtpFile)
      .then(res => res.arrayBuffer())
      .then(buffer => {
        reader.parseAsArrayBuffer(buffer);
        const polydata = reader.getOutputData();

        const mapper = vtk.Rendering.Core.vtkMapper.newInstance();
        mapper.setInputData(polydata);

        const scalars = polydata.getPointData().getScalars();

        if (scalars) {
          const lut = vtk.Rendering.Core.vtkColorTransferFunction.newInstance();

          // Load colormap and build LUT
          if (scalarType === 'discrete') {
          fetch('assets/data/colormap.txt')
            .then(res => res.text())
            .then(text => {
              const lines = text.trim().split('\n');
              lines.forEach(line => {
                const [row, cls, r, g, b, a] = line.trim().split(/\s+/).map(Number);
                lut.addRGBPoint(cls, r / 255, g / 255, b / 255);
              });

              // Apply LUT and mapper settings
              mapper.setLookupTable(lut);
              mapper.setColorModeToMapScalars();
              mapper.setScalarModeToUsePointData();
              mapper.setScalarVisibility(true);

              // Set scalar range to cover all classes
              mapper.setScalarRange(0, 32);

              renderWindow.render();
            });
          }
          else if (scalarType === 'continuous')
          {
          // Helper: convert hex to [r,g,b] in 0–1
          function hexToRgb01(hex) {
            const bigint = parseInt(hex.slice(1), 16);
            const r = ((bigint >> 16) & 255) / 255.0;
            const g = ((bigint >> 8) & 255) / 255.0;
            const b = (bigint & 255) / 255.0;
            return [r, g, b];
          }
          const lut = vtk.Rendering.Core.vtkColorTransferFunction.newInstance();

          lut.setNanColor(0.5, 0.5, 0.5, 1.0);
          lut.setBelowRangeColor(hexToRgb01("#82cc12"));
          lut.setAboveRangeColor(hexToRgb01("#c05eeb"));
          lut.setUseBelowRangeColor(true);
          lut.setUseAboveRangeColor(true);

            // Define cvals and hex colors
            const cvals = [1.5, 2, 3, 4, 5, 5.5];
            const colors = ["#a1cc12", "#faf561", "#fadf69", "#fa9e61", "#f75757", "#c05eeb"];

            // Add color points
            cvals.forEach((val, i) => {
              const [r, g, b] = hexToRgb01(colors[i]);
              lut.addRGBPoint(val, r, g, b);
            });

            // Apply LUT and mapper settings
            mapper.setLookupTable(lut);
            mapper.setColorModeToMapScalars();
            mapper.setScalarModeToUsePointData();
            mapper.setScalarVisibility(true);

            // Scalar range should match your cvals domain
            mapper.setScalarRange(0.4, 2.0);

            renderWindow.render();
          }
        }

        const actor = vtk.Rendering.Core.vtkActor.newInstance();
        actor.setMapper(mapper);

        // Actor appearance
        actor.getProperty().setAmbient(0.2);
        actor.getProperty().setDiffuse(0.7);
        actor.getProperty().setSpecular(0.3);
        actor.getProperty().setSpecularPower(10.0);

        renderer.addActor(actor);
        renderer.resetCamera();

        const camera = renderer.getActiveCamera();
        camera.elevation(90);
        renderer.resetCameraClippingRange();

        // Initial render (in case LUT is still loading)
        renderWindow.render();
      })
      .catch(err => console.error('Failed to load VTP:', err));
  });
}
