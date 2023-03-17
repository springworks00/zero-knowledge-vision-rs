// consider draw_matches()

use opencv::{
    core::{self, KeyPoint, Mat},
    features2d::{self, ORB},
    xfeatures2d,
};
use ndarray::Array2;
use std::iter::repeat_with;
//use kmeans::{KMeans, KMeansBuilder};
use rand::SeedableRng;
use rand::rngs::StdRng;

use opencv::core::DMatch;
use opencv::features2d::BFMatcher;

use crate::util::{Frame, vec_to_vector_keypoint};
use opencv::prelude::DescriptorMatcher;
use opencv::prelude::Feature2DTrait;
use opencv::core::Vector;
use opencv::core::MatTraitConst;
use opencv::prelude::MatTraitConstManual;
use ndarray::stack;

use std::io::Write;

type Result<T> = std::result::Result<T, opencv::Error>;

pub fn orb_features(frames: &mut Vec<Frame>, n: usize) -> Result<()> {
    let mut orb = <dyn ORB>::create(
        n as i32, // nfeatures
        1.2, // scaleFactor
        8, // nlevels
        31, // edgeThreshold
        0, // firstLevel
        2, // WTA_K
        features2d::ORB_ScoreType::HARRIS_SCORE, // scoreType
        31, // patchSize
        20, // fastThreshold
    )?;
   
    for frame in frames.iter_mut() {
        let mut kps = Vector::new();
        let mut dcs = Mat::default();
        
        orb.detect_and_compute(&frame.raw, &Mat::default(), &mut kps, &mut dcs, false)?;
        frame.kps = Some(kps);
        frame.dcs = Some(dcs);
    }
    
    Ok(())
}

use opencv::prelude::BinaryDescriptorMatcherTraitConst;

pub fn most_similar(scan: &Vec<Frame>, x: &Frame, threshold: f32) -> Result<usize> {
    let bf_matcher = opencv::line_descriptor::BinaryDescriptorMatcher::default()?;
    let mut best_match_index = 0;
    let mut best_match_count = 0;

    for (index, frame) in scan.iter().enumerate() {
        let mut matches = Vector::new();
        bf_matcher.match_(
            &x.dcs.as_ref().unwrap(), 
            &frame.dcs.as_ref().unwrap(), 
            &mut matches, 
            &Mat::default(),
        )?;

        // Count good matches
        let good_matches = matches.iter().filter(|m| m.distance < threshold).count();

        // Update best match
        if good_matches > best_match_count {
            best_match_count = good_matches;
            best_match_index = index;
        }
    }

    Ok(best_match_index)
}
//use ndarray::Array2;
/*
fn ndarray_to_mat(array: &Array2<u8>) -> Result<Mat> {
    let rows = array.nrows() as i32;
    let cols = array.ncols() as i32;

    let mut mat = Mat::zeros(rows, cols, opencv::core::CV_8U)?;

    for r in 0..rows {
        for c in 0..cols {
            let value = array[(r as usize, c as usize)];
            let row_ptr = mat.ptr_mut::<u8>(r)?;
            row_ptr[c as usize] = value;
        }
    }

    Ok(mat)
}
*/
/*
fn _match(query: &Frame, train: &Frame, threshold: f32) -> Result<Vec<DMatch>> {
    let matcher = BFMatcher::new(opencv::core::NORM_HAMMING, false)?;
    let mut matches = opencv::types::VectorOfVectorOfDMatch::new();

    matcher.knn_match(
        &query.dcs, //.as_ref().unwrap(),
        //&train.dcs.as_ref().unwrap(), 
        &mut matches, 
        2, 
        &Mat::default(), 
        false,
    )?;

    let mut good_matches = Vec::new();
    for i in 0..matches.len() {
        let match_pair = matches.get(i)?;
        if match_pair.len() == 2 {
            let m = DMatch::from(match_pair.get(0)?);
            let n = DMatch::from(match_pair.get(1)?);

            if m.distance < threshold * n.distance {
                good_matches.push(m);
            }
        }
    }

    Ok(good_matches)
}
*/
/*
use ndarray::prelude::*;
use ndarray::Axis;

fn _acc_match(query: &Array2<u8>, train: &mut Array2<u8>) -> Vec<usize> { //, Array2<u8>) {
    let matcher = features2d::BFMatcher::new(core::NORM_HAMMING, false).unwrap();
    let mut matches = opencv::types::VectorOfVectorOfDMatch::new();

    matcher.knn_match(
        query, //.view(), //.into_dyn().into_owned(),
        //train.view().into_dyn().into_owned(),
        &mut matches, 
        2, 
        &Mat::default(), 
        false,
    ).unwrap();

    let (mut good, mut bad) = (Vec::new(), Vec::new());
    for m_n in matches {
        let m_n0 = DMatch::from(m_n.get(0).unwrap());
        let m_n1 = DMatch::from(m_n.get(1).unwrap());
        if m_n0.distance < 0.75 * m_n1.distance {
            good.push(m_n1.train_idx as usize);
        } else {
            bad.push(m_n0.query_idx as usize);
        }
    }

    for x in bad.iter().map(|&idx| query.slice(ndarray::s![idx, ..])).collect::<Vec<_>>() {
        /*
        train = ndarray::stack(
            ndarray::Axis(0), 
            &[train.view(), x.view()],
        ).unwrap();
        */
        // *train = stack![
        //    Axis(0),
        *train = [train, &mut x.to_owned().into_shape((1, x.len())).unwrap()];
            //&mut x.to_owned().into_shape((1, x.len())).expect("Failed to reshape array")]
    }

    good
}
*/
/*
fn cluster(frames: &[Frame], k: usize) -> Result<Vec<Frame>> {
    let dcss: Vec<_> = frames.iter().map(|f| f.dcs.as_ref().unwrap().clone()).collect();

    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut train = dcss[0].clone();

    let len_dcss = dcss.len();
    for (frame_idx, dcs) in dcss.iter().enumerate() {
        let (this_ys, new_train) = _acc_match(dcs, &mut train);
        let this_xs = repeat_with(|| frame_idx).take(this_ys.len()).collect::<Vec<_>>();

        ys.extend(this_ys);
        xs.extend(this_xs);
        status(&format!("building cluster set: {}%", 100 * (frame_idx + 1) / len_dcss));
    }
    println!();

    status(&format!("clustering to {} frames: ", k));

    let points: Vec<[usize; 2]> = xs.into_iter().zip(ys.into_iter()).map(|(x, y)| [x, y]).collect();
    let mut rng = StdRng::seed_from_u64(1);
    let kmeans = KMeansBuilder::<[usize; 2], _>::new(k)
        .random_initialization()
        .build_with_rng(&mut rng)
        .unwrap();

    let (kmeans, _) = kmeans.fit(&points);
    let centroids = kmeans.centroids();

    let mut best_indexes = vec![0; k];
    for (i, point) in points.iter().enumerate() {
        let (closest_index, _) = centroids.iter().enumerate().min_by_key(|&(_, c)| c.squared_distance(point)).unwrap();
        best_indexes[closest_index] = i;
    }

    println!("100%");
    Ok(best_indexes.into_iter().map(|i| frames[points[i][0]].clone()).collect())
}
*/
