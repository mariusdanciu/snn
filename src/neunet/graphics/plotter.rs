use plotters::prelude::*;

pub fn plot_data(file: String, train: &Vec<f32>, test: &Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
    fn with_index(v: &Vec<f32>) -> Vec<(f32, f32)> {
        let mut idx = -1;

        let r: Vec<(f32, f32)> = v.iter().map(|e| {
            idx += 1;
            (idx as f32, *e)
        }).collect();

        r
    }
    let root_area = BitMapBackend::new(&file, (1024, 768)).into_drawing_area();

    root_area.fill(&WHITE)?;

    let root_area = root_area.titled("Train vs. Test Accuracy", ("sans-serif", 60).into_font())?;

    let mut cc = ChartBuilder::on(&root_area)
        .margin(40)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_ranged(0f32..((train.len() - 1) as f32), 0f32..1f32)?;

    cc.configure_mesh()
        .x_labels(20)
        .x_label_formatter(&|v| format!("{:.1}", v))
        .y_label_formatter(&|v| format!("{:.1}", v))
        .draw()?;

    cc.draw_series(LineSeries::new(
        with_index(&train),
        &RED,
    ))?
        .label("Train")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    cc.draw_series(LineSeries::new(
        with_index(&test),
        &BLUE,
    ))?
        .label("Test")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    cc
        .configure_series_labels()
        .background_style(&WHITE.mix(0.5))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    plot_data(String::from("./sample.png"), &vec![0.34, 0.48, 0.489, 0.43], &vec![0.43, 0.45, 0.39, 0.63])
}
