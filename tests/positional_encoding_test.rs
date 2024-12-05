use Transformer::math::positional_encoding::sinusoidal_pos_encoding;


    #[test]
    fn test_sin_encoding_even_index() {
        let pos = 1;
        let index = 2; // even index
        let embedding_size = 512;
        let result = sinusoidal_pos_encoding(pos, index, embedding_size);
        // Expected result: sin(1 / (10000^(2*index/embedding_size)))
        let divisor = 10000f32.powf(2.0 * (index as f32 / embedding_size as f32));
        let expected = (pos as f32 / divisor).sin();
        assert!((result - expected).abs() < 1e-6, "Failed for even index");
    }

    #[test]
    fn test_cos_encoding_odd_index() {
        let pos = 1;
        let index = 3; // odd index
        let embedding_size = 512;
        let result = sinusoidal_pos_encoding(pos, index, embedding_size);
        // Expected result: cos(1 / (10000^(2*index/embedding_size)))
        let divisor = 10000f32.powf(2.0 * (index as f32 / embedding_size as f32));
        let expected = (pos as f32 / divisor).cos();
        assert!((result - expected).abs() < 1e-6, "Failed for odd index");
    }

    #[test]
    fn test_large_position() {
        let pos = 1000;
        let index = 2;
        let embedding_size = 512;
        let result = sinusoidal_pos_encoding(pos, index, embedding_size);
        let divisor = 10000f32.powf(2.0 * (index as f32 / embedding_size as f32));
        let expected = (pos as f32 / divisor).sin();
        assert!((result - expected).abs() < 1e-6, "Failed for large position");
    }

    #[test]
    fn test_zero_position() {
        let pos = 0;
        let index = 1;
        let embedding_size = 512;
        let result = sinusoidal_pos_encoding(pos, index, embedding_size);
        assert_eq!(result, 0.0, "Failed for zero position");
    }

    #[test]
    fn test_large_embedding_size() {
        let pos = 5;
        let index = 10;
        let embedding_size = 2048;
        let result = sinusoidal_pos_encoding(pos, index, embedding_size);
        let divisor = 10000f32.powf(2.0 * (index as f32 / embedding_size as f32));
        let expected = (pos as f32 / divisor).cos();
        assert!((result - expected).abs() < 1e-6, "Failed for large embedding size");
    }
