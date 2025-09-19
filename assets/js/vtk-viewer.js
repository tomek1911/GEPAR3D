// assets/js/vtk-viewer.js

function initVTKViewer(containerId, vtpFile) {
  window.requestAnimationFrame(() => {
    const container = document.querySelector(containerId);

    // Set container size
    const width = window.innerWidth * 0.33;
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
